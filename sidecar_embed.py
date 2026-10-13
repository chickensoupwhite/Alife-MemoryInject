# -*- coding: utf-8 -*-
"""Mao.MemoryRecall —— 向量化服务 + 记忆召回（常驻子进程）。

协议：stdin 每行一个 JSON 请求，stdout 每行一个 JSON 响应。
  {"id":1,"fn":"embed","texts":["...","..."]}     -> {"id":1,"ok":true,"vectors":[[...],...]}
  {"id":2,"fn":"ping"}                             -> {"id":2,"ok":true,"device":"cuda","dim":512}
  {"id":3,"fn":"recall","query":"...","top_k":5}   -> {"id":3,"ok":true,"bookmarks":"..."}
  {"fn":"__shutdown__"}                            -> 退出

embed 用模型：BAAI/bge-small-zh-v1.5（512 维，中文短文本）。
recall  纯标准库：读 index.json，按 人/事/物 标签打分，输出书签文本。
        ★add-only 原则：命中只往消息末尾"追加"提示，绝不改写历史/不塞整句，
          以免击穿前缀缓存（参考 livingmemory 的 extra_user_content 注入）。
两者互不依赖——模型没就绪时 recall 照常可用；索引没就绪时 embed 照常可用。
"""
import sys
import os
import io
import re
import json
import datetime

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def find_model_dir():
    env = os.environ.get("MAO_EMBED_MODEL")
    if env and os.path.isdir(env):
        return env
    cache = os.environ.get("MODELSCOPE_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "modelscope", "hub")
    cands = [
        os.path.join(cache, "models", "BAAI", "bge-small-zh-v1.5"),
        os.path.join(cache, "models", "BAAI", "bge-small-zh-v1___5"),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise RuntimeError("未找到 bge-small-zh-v1.5 模型目录：" + cache)


# ============================================================
# 记忆召回（纯标准库，不依赖 torch）
# 打分逻辑与 recall_engine.py 同源：标签(人/事/物) > 标题 > 摘要 > 正文。
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))

W_TAG = 3      # people / topics / things 标签命中
W_HEAD = 3     # 块标题命中
W_SUMMARY = 2  # 首句摘要命中
W_TEXT = 1     # 正文命中

CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
WORD_RE = re.compile(r"[A-Za-z0-9]+")

_IDX = {"path": None, "data": None, "vocab": None, "df": None}


def _find_index():
    """找 index.json：先看脚本同目录，再看上级目录。"""
    for c in (os.path.join(_HERE, "index.json"),
              os.path.join(os.path.dirname(_HERE), "index.json")):
        if os.path.isfile(c):
            return c
    return None


def _set_index(path):
    """切换当前使用的索引文件（多角色各自一份，不打架）。"""
    if path and os.path.isfile(path) and _IDX.get("path") != path:
        _IDX["path"] = path
        _IDX["data"] = None
        _IDX["vocab"] = None
        _IDX["df"] = None


def _load_index():
    """加载索引（带缓存）。找不到就返回空表，不抛异常。"""
    if _IDX["data"] is None:
        p = _IDX["path"] or _find_index()
        _IDX["path"] = p
        if p:
            with open(p, encoding="utf-8") as f:
                _IDX["data"] = json.load(f)
        else:
            _IDX["data"] = []
    return _IDX["data"]


def _vocab():
    """从索引收集全部标签词（人/事/物），长词优先，供最长匹配切词。"""
    if _IDX["vocab"] is None:
        words = set()
        for e in _load_index():
            for field in ("people", "topics", "things"):
                for w in e.get(field) or []:
                    if w:
                        words.add(w)
        # 长的排前面：保证"血压计"不会被切成"血压"
        _IDX["vocab"] = sorted(words, key=lambda w: -len(w))
    return _IDX["vocab"]


def tokenize(query):
    """把一句中文切成查询词。标签词最长匹配 + 汉字 2 字滑窗 + 英文串。"""
    if not query:
        return []
    q = query.lower()
    terms = []
    used = []

    def covered(pos):
        return any(a <= pos < b for a, b in used)

    # 1) 标签词最长匹配
    for w in _vocab():
        lw = w.lower()
        start = 0
        while True:
            p = q.find(lw, start)
            if p < 0:
                break
            start = p + 1
            if covered(p):
                continue
            terms.append(lw)
            used.append((p, p + len(lw)))
    # 2) 汉字串 2 字滑窗
    for m in CJK_RUN_RE.finditer(q):
        run = m.group()
        base = m.start()
        if len(run) == 1:
            if not covered(base):
                terms.append(run)
            continue
        # 短串整体成词：提高精度（"帕尔曼"整体命中 > 拆成"帕尔"+"尔曼"）
        if len(run) <= 8 and not covered(base):
            terms.append(run)
            # 整串已占位：下面的 2 字滑窗要跳过这段，否则还会吐碎片噪声
            used.append((base, base + len(run)))
        for i in range(len(run) - 1):
            pos = base + i
            if covered(pos) and covered(pos + 1):
                continue
            terms.append(run[i:i + 2])
    # 3) 英文/数字串
    for m in WORD_RE.finditer(q):
        if not covered(m.start()):
            terms.append(m.group().lower())

    seen = set()
    out = []
    for t in terms:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _field_text(e, field):
    v = e.get(field) or ""
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v)


def score_entry(e, terms):
    """按权重累加打分；返回 (总分, 命中词数)。"""
    tags = " ".join([_field_text(e, "people"), _field_text(e, "topics"), _field_text(e, "things")]).lower()
    head = _field_text(e, "head").lower()
    summary = _field_text(e, "summary").lower()
    text = _field_text(e, "text").lower()
    total = 0
    hit = 0
    for t in terms:
        if t in tags:
            s = W_TAG
        elif t in head:
            s = W_HEAD
        elif t in summary:
            s = W_SUMMARY
        elif t in text:
            s = W_TEXT
        else:
            s = 0
        if s:
            total += s
            hit += 1
    return total, hit


def _df():
    """每个词在多少条里出现过（文档频率），用来给常见词降权。"""
    if _IDX.get("df") is not None:
        return _IDX["df"]
    df = {}
    for e in _load_index():
        tags = " ".join([_field_text(e, "people"), _field_text(e, "topics"),
                         _field_text(e, "things"), _field_text(e, "head"),
                         _field_text(e, "summary"), _field_text(e, "text")]).lower()
        for t in set(tokenize(tags)):
            df[t] = df.get(t, 0) + 1
    _IDX["df"] = df
    return df


def _idf(term, n):
    """常见词权重小（"真央"满篇都是就没区分度），稀罕词权重大。"""
    import math
    df = _df().get(term, 0)
    return math.log((n + 1.0) / (df + 1.0)) + 1.0


def rel_time(date_str, today=None):
    """相对时间：今天/昨天/N天前/上周/N周前/N个月前/N年前。读取时按当下重算。"""
    try:
        d = datetime.date.fromisoformat(str(date_str)[:10])
    except Exception:
        return str(date_str)
    today = today or datetime.date.today()
    n = (today - d).days
    if n < 0:
        return str(date_str)
    if n == 0:
        return "今天"
    if n == 1:
        return "昨天"
    if n < 7:
        return "%d天前" % n
    if n < 14:
        return "上周"
    if n < 60:
        return "%d周前" % (n // 7)
    if n < 365:
        return "%d个月前" % (n // 30)
    return "%d年前" % (n // 365)


def parse_time_hint(text):
    """把中文时间词解析成 (date_from, date_to, period)。解析不出就返回 (None,None,None)。
    只做保守解析：识别不出宁可不过滤，避免误杀。"""
    if not text:
        return None, None, None
    t = str(text)
    today = datetime.date.today()
    df = dt = None
    period = None

    for kw, p in (("凌晨", "凌晨"), ("早上", "早上"), ("早晨", "早上"), ("上午", "上午"),
                  ("中午", "中午"), ("午后", "下午"), ("下午", "下午"),
                  ("傍晚", "傍晚"), ("晚上", "晚上"), ("夜里", "夜"), ("深夜", "深夜"), ("晚", "晚")):
        if kw in t:
            period = p
            break

    m = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", t)
    if m:
        df = dt = "%04d-%02d-%02d" % (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    else:
        m = re.search(r"(\d{1,2})月(\d{1,2})[日号]", t)
        if m:
            df = dt = "%04d-%02d-%02d" % (today.year, int(m.group(1)), int(m.group(2)))
        elif "前天" in t:
            df = dt = (today - datetime.timedelta(days=2)).isoformat()
        elif "昨天" in t or "昨晚" in t:
            df = dt = (today - datetime.timedelta(days=1)).isoformat()
        elif "今天" in t or "今日" in t or "今晚" in t or "刚才" in t:
            df = dt = today.isoformat()
        else:
            m = re.search(r"(\d+)\s*天前", t)
            if m:
                df = dt = (today - datetime.timedelta(days=int(m.group(1)))).isoformat()
            elif "上周" in t:
                df = (today - datetime.timedelta(days=today.weekday() + 7)).isoformat()
                dt = (today - datetime.timedelta(days=today.weekday() + 1)).isoformat()
            elif "这周" in t or "本周" in t:
                df = (today - datetime.timedelta(days=today.weekday())).isoformat()
                dt = today.isoformat()
            elif "上个月" in t:
                df = (today.replace(day=1) - datetime.timedelta(days=1)).replace(day=1).isoformat()
                dt = (today.replace(day=1) - datetime.timedelta(days=1)).isoformat()
    return df, dt, period


def recall(query, top_k=5, min_ratio=0.45, date_from=None, date_to=None, period=None,
           hour_from=None, hour_to=None, use_graph=True):
    """
    召回：输入一句中文，返回命中的记忆条目（按相关度倒序）。
    打分 = 字段权重（标签3>标题3>摘要2>正文1）× 词权重（IDF，常见词降权）。
    过滤：①时间维过滤（date_from/date_to/period）②尾部噪声 ③无命中返回 []。
    """
    if date_from is None and date_to is None and period is None:
        hdf, hdt, hp = parse_time_hint(query)
        date_from = date_from or hdf
        date_to = date_to or hdt
        period = period or hp
    terms = tokenize(query)
    if not terms:
        return []
    data = [e for e in _load_index() if not e.get("retired")]
    if date_from or date_to or period or hour_from is not None or hour_to is not None:
        def _ok(e):
            d = str(e.get("date") or "")
            if date_from and d and d < date_from:
                return False
            if date_to and d and d > date_to:
                return False
            if period and period not in str(e.get("period") or ""):
                return False
            if hour_from is not None or hour_to is not None:
                h = e.get("hour")
                if h is not None:
                    if hour_from is not None and h < hour_from:
                        return False
                    if hour_to is not None and h > hour_to:
                        return False
            return True
        data = [e for e in data if _ok(e)]
    n = len(data) or 1
    scored = []
    for e in data:
        raw, hit = score_entry(e, terms)
        if raw <= 0 or hit == 0:
            continue
        tags = " ".join([_field_text(e, "people"), _field_text(e, "topics"),
                         _field_text(e, "things")]).lower()
        head = _field_text(e, "head").lower()
        summary = _field_text(e, "summary").lower()
        text = _field_text(e, "text").lower()
        w = 0.0
        for t in terms:
            if t in tags:
                fw = W_TAG
            elif t in head:
                fw = W_HEAD
            elif t in summary:
                fw = W_SUMMARY
            elif t in text:
                fw = W_TEXT
            else:
                continue
            w += fw * _idf(t, n)
        scored.append((w, hit, e.get("time") or "", e))
    if not scored:
        # 关键词/语义都没命中：试图谱路兜底（顺关系链捞）
        gexp = graph_expand(query, terms) if use_graph else []
        if gexp:
            for e in data:
                etags = " ".join([_field_text(e, "people"), _field_text(e, "topics"),
                                  _field_text(e, "things")]).lower()
                gh = [g for g in gexp if g.lower() in etags]
                if gh:
                    scored.append((0.2 * len(gh), len(gh), e.get("time") or "", e))
        if not scored:
            return []
    # 关键词路：字段权重×IDF，降序 → 排名
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    kw_rank = {id(x[3]): i + 1 for i, x in enumerate(scored)}
    kw_map = {id(x[3]): x for x in scored}

    # 语义路：向量余弦，降序 → 排名（拿不到向量就整体退回纯词频）
    qv = _embed([query]) if _MODEL["ready"] else None
    fused = None
    if qv:
        sims = []
        for _, _, _, e in scored:
            ev = _entry_vector(e)
            if ev:
                sims.append((_cos(qv[0], ev), e))
        if sims:
            sims.sort(key=lambda x: x[0], reverse=True)
            sem_rank = {id(e): i + 1 for i, (_, e) in enumerate(sims)}
            K = 60
            fuse = []
            for _, _, _, e in scored:
                rrf = 0.0
                if id(e) in kw_rank:
                    rrf += 1.0 / (K + kw_rank[id(e)])
                if id(e) in sem_rank:
                    rrf += 1.0 / (K + sem_rank[id(e)])
                dual = id(e) in kw_rank and id(e) in sem_rank
                # 强化 + 衰减：命中越多越上浮，越久没碰越淡（上限 +25%，别压过 RRF 主体）
                boost = _entry_weight(e)
                fuse.append((rrf * boost, 1 if dual else 0, e.get("time") or "", e))
            fuse.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            fused = fuse

    final = fused if fused else [
        (w / 1000.0 * _entry_weight(e), hit, t, e)
        for w, hit, t, e in scored]
    top = final[0][0]
    cutoff = top * min_ratio
    out = []
    for rrf, _, _, e in final:
        if rrf < cutoff:
            break
        item = dict(e)
        item["rel_time"] = rel_time(e.get("date", ""))
        _touch(e)   # 生命周期：命中即计一次访问（写回有节流）
        out.append(item)
        if len(out) >= top_k:
            break
    return out


# ===== 记忆生命周期：访问计数 + 强化（写回节流） =====

_LIFE = {"dirty": False, "last_write": 0.0}


def _touch(entry):
    """命中一条：访问次数+1、更新 last_access（相对时间会被重算）。"""
    try:
        entry["access_count"] = int(entry.get("access_count") or 0) + 1
        entry["last_access"] = datetime.date.today().isoformat()
        _LIFE["dirty"] = True
        _flush_life()
    except Exception:
        pass


def _flush_life():
    """节流写回：20 秒内最多落盘一次。任何异常都不外抛。"""
    import time as _t
    if not _LIFE["dirty"]:
        return
    now = _t.time()
    if now - _LIFE["last_write"] < 20:
        return
    p = _IDX.get("path") or _find_index()
    if not p:
        return
    try:
        io.open(p, "w", encoding="utf-8").write(
            json.dumps(_IDX["data"], ensure_ascii=False, indent=1))
        _LIFE["last_write"] = now
        _LIFE["dirty"] = False
    except Exception:
        pass


# ===== 记忆生命周期：衰减淡出（半衰期）+ 软删→物理清 =====

_CFG = {"half_life": 30, "block_max": 200}   # 运行时可配（由插件配置传入）

DECAY_HALF_LIFE_DAYS = 30      # 半衰期：30 天没碰，权重减半（默认值，可被 _CFG 覆盖）
DECAY_FLOOR = 0.05             # 衰减下限，别归零（怕误杀重要旧账）
SOFT_DELETE_DAYS = 60          # 从没访问过且老过这么久 → 才考虑软删（门槛要高，防误杀）
HARD_DELETE_DAYS = 30          # 软删后再过这么久 → 物理清


def _decay_factor(entry):
    """按半衰期算有效权重系数（0.05~1.0）。访问过就按 last_access 起算，等于强化归零。"""
    base = entry.get("last_access") or entry.get("date") or ""
    try:
        d = datetime.date.fromisoformat(str(base)[:10])
    except Exception:
        return 1.0
    age = (datetime.date.today() - d).days
    if age <= 0:
        return 1.0
    import math
    f = 0.5 ** (age / float(_CFG.get("half_life") or DECAY_HALF_LIFE_DAYS))
    return max(DECAY_FLOOR, min(1.0, f))


def _entry_weight(entry):
    """综合强化(access_count)与衰减(decay)的排序加成系数。上限 +25%。"""
    ac = min(int(entry.get("access_count") or 0), 5)
    reinforce = 1.0 + 0.05 * ac
    decay = _decay_factor(entry)
    # 强化过的不该被时间压死：以两者乘积，但衰减只压到 reinforce 的一半强度
    return reinforce * (0.5 + 0.5 * decay)


def _looks_important(e):
    """像牵挂/欠账/约定/健康这类，永远不软删。"""
    blob = (_field_text(e, "head") + _field_text(e, "summary") + _field_text(e, "text"))
    for k in ("牵挂", "欠账", "约定", "待办", "血压", "禁欲", "早睡", "欠", "别忘"):
        if k in blob:
            return True
    return False


def maintain_index(memdir=None, now=None, allow_soft=False):
    """每日维护：软删(标记 retired) → 物理清。带"错过天数补偿"——久没跑一次跑到位。
    ★软删默认关闭（allow_soft=False）：只有显式开启，才把"从未访问 + 老过60天 +
      不像牵挂/约定"的条目标记 retired；物理清只针对已 retired 且够久的。
    返回 {scanned, soft_deleted, purged}。不改文件，只改 index.json。"""
    data = _load_index()
    today = now or datetime.date.today()
    soft = purged = 0
    keep = []
    for e in data:
        # 已经软删的：看多久了，够久就物理清
        if e.get("retired"):
            try:
                rd = datetime.date.fromisoformat(str(e.get("retired_at"))[:10])
            except Exception:
                rd = today
            if (today - rd).days >= HARD_DELETE_DAYS:
                purged += 1
                continue
            keep.append(e)
            continue
        # 判定软删：从没被访问过、且老过 SOFT_DELETE_DAYS、且衰减很低
        last = e.get("last_access") or ""
        ac = int(e.get("access_count") or 0)
        try:
            d = datetime.date.fromisoformat(str(e.get("date") or "")[:10])
        except Exception:
            d = today
        age = (today - d).days
        if allow_soft and ac == 0 and age >= SOFT_DELETE_DAYS and not last and not _looks_important(e):
            e["retired"] = True
            e["retired_at"] = today.isoformat()
            soft += 1
        keep.append(e)
    if soft or purged:
        p2 = _IDX.get("path") or _find_index()
        if p2:
            io.open(p2, "w", encoding="utf-8").write(json.dumps(keep, ensure_ascii=False, indent=1))
            _IDX["data"] = None
    return {"scanned": len(data), "soft_deleted": soft, "purged": purged}


# ===== 图谱路：人/事/物 节点 + 共现边，顺关系链扩一层 =====

_GRAPH = {"nodes": {}, "edges": {}, "built": False}


def build_graph():
    """从 index 抽节点（人/事/物/高频实词）与共现边。
    节点=出现在条目里的标签词；边=同一条目内两节点共现（权重=共现次数）。
    """
    nodes = {}
    edges = {}
    for e in _load_index():
        if e.get("retired"):
            continue
        ns = []
        for field in ("people", "topics", "things"):
            for x in (e.get(field) or []):
                if x:
                    ns.append(str(x))
        ns = list(dict.fromkeys(ns))
        for n in ns:
            nodes[n] = nodes.get(n, 0) + 1
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                a, b = sorted((ns[i], ns[j]))
                k = a + "\t" + b
                edges[k] = edges.get(k, 0) + 1
    _GRAPH.update({"nodes": nodes, "edges": edges, "built": True})
    return len(nodes), len(edges)


def _neighbors(node, top=6):
    """取某节点最相关的邻居（按共现权重），返回 [(邻居, 权重)]。"""
    if not _GRAPH["built"]:
        build_graph()
    out = []
    for k, w in _GRAPH["edges"].items():
        a, b = k.split("\t")
        if a == node:
            out.append((b, w))
        elif b == node:
            out.append((a, w))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top]


def graph_expand(query, terms, top=6):
    """图谱扩词：query/terms 命中的节点，取它们的邻居，返回扩展词集。
    只扩一层，权重低，仅作第三路补充，不压关键词/语义两路。"""
    if not _GRAPH["built"]:
        build_graph()
    seed = []
    ql = (query or "").lower()
    for n in _GRAPH["nodes"]:
        nl = n.lower()
        if nl in ql or any(nl in t or t in nl for t in terms if len(t) >= 2):
            seed.append(n)
    if not seed:
        return []
    expanded = {}
    for s in seed[:4]:
        for nb, w in _neighbors(s):
            if nb not in seed:
                expanded[nb] = expanded.get(nb, 0) + w
    ranked = sorted(expanded.items(), key=lambda x: x[1], reverse=True)
    return [n for n, _ in ranked[:top]]


# ===== 可视化：把索引导成一页 HTML（本地打开看） =====

def render_view(outpath=None):
    """生成索引浏览页：每条 相对时间/人事物/摘要/出处，按时间倒序。"""
    import html
    data = _load_index()
    data = [e for e in data if not e.get("retired")]
    data.sort(key=lambda e: e.get("time") or "", reverse=True)
    rows = []
    for e in data:
        tags = " ".join(["#" + x for x in (e.get("people") or []) + (e.get("topics") or []) + (e.get("things") or [])])
        ac = int(e.get("access_count") or 0)
        star = "★%d" % ac if ac else ""
        rows.append(
            '<div class="c"><div class="m">[%s] %s %s</div><div class="t">%s</div>'
            '<div class="s">%s</div><div class="o">%s</div></div>' % (
                html.escape(e.get("rel_time") or rel_time(e.get("date", ""))),
                html.escape(tags), star,
                html.escape((e.get("head") or "").lstrip("#").strip()[:60]),
                html.escape((e.get("summary") or e.get("text") or "")[:120]),
                html.escape(e.get("source") or "-")))
    doc = ("<!doctype html><meta charset=utf-8><title>记忆索引</title>"
           "<style>body{font-family:system-ui;background:#1b1b1f;color:#ddd;margin:0;padding:16px}"
           ".c{border-left:3px solid #d97757;background:#26262b;margin:6px 0;padding:8px 12px;border-radius:6px}"
           ".m{color:#d97757;font-size:12px}.t{font-weight:600;margin:2px 0}.s{color:#aaa;font-size:13px}"
           ".o{color:#666;font-size:11px;margin-top:2px}</style>"
           "<h2>记忆索引（%d 条）</h2>" % len(data) + "".join(rows))
    if not outpath:
        outpath = os.path.join(_HERE, "记忆索引视图.html")
    io.open(outpath, "w", encoding="utf-8").write(doc)
    return len(data), outpath


def to_bookmarks(hits):
    """生成注入用的书签文本，每行一条；无命中返回空串。"""
    if not hits:
        return ""
    lines = []
    for e in hits:
        label = "·".join([t for t in (e.get("topics") or []) + (e.get("things") or []) if t])
        if not label:
            label = (e.get("head") or "记忆").lstrip("#").strip() or "记忆"
        summary = (e.get("summary") or "").strip()
        if not summary:
            summary = (e.get("text") or "").strip()[:40]
        lines.append("- [%s] %s：%s（出处 %s）" % (
            e.get("rel_time") or rel_time(e.get("date", "")),
            label, summary, e.get("source") or "-"))
    return "\n".join(lines)


# ============================================================
# 主循环
# ============================================================

# ============================================================
# 索引构建（从 tag_index.py 合并进来，内置，不依赖外部脚本）
# 这是"人/事/物"打标规则的唯一出处，改规则只改这里。
# ============================================================

PEOPLE = {
    "主人": ["主人", "570167521", "白姬汤"],
    "小鲸鱼": ["小鲸鱼", "鲸鱼娘"],
    "俾斯麦": ["俾斯麦"],
    "妈": ["妈妈", "他妈", "母亲", "咱妈"],
    "麦乐迪": ["麦乐迪"],
}
TOPICS = {
    "健康": ["血压", "低压", "高压", "睡眠", "早睡", "饮食", "头痛", "头晕", "耳鸣", "减肥", "禁欲", "体重", "运动", "吃药", "晕厥"],
    "学习": ["综应", "职测", "资料分析", "案例分析", "作文", "背诵", "考点", "速算", "逻辑", "常识", "辨析", "讲义", "真题", "笔记", "班会", "研学"],
    "技术": ["插件", "脚本", "ComfyUI", "模型", "配置", "重启", "代码", "索引", "端口", "内存", "显存", "工作流", "regedit", "注册表"],
    "约定": ["约定", "陪跑", "记账", "提醒", "打卡"],
    "关系": ["晚安", "摸头", "工具人", "名字", "重要", "真央", "立绘"],
    "生活": ["吃饭", "天气", "衣服", "Steam", "游戏", "群聊", "早餐", "午餐", "晚餐"],
}
THINGS = {
    "血压计": ["血压计", "血压"],
    "粉笔": ["粉笔"],
    "ComfyUI": ["ComfyUI", "comfyui"],
    "Obsidian": ["Obsidian", "obsidian"],
    "QQ": ["QQ", "qchat", "QuickChat"],
    "网易云": ["网易云"],
    "MC": ["MC", "我的世界", "minecraft"],
    "街霸": ["街霸"],
    "桌宠": ["桌宠"],
    "屏幕": ["屏幕"],
    "CodeBuddy": ["CodeBuddy", "codebuddy", "buddy"],
    "记忆插件": ["MemoryInject", "记忆插件", "RecallHint"],
}
PERIOD_HOUR = {"凌晨": 3, "早上": 8, "早晨": 8, "上午": 10, "中午": 12, "午后": 14,
               "下午": 15, "傍晚": 18, "晚": 20, "晚上": 21, "夜": 22, "深夜": 23}


def _split_fine(head, body):
    """把一个大标题块切细：列表项/空行/句末切，单块上限 200 字。
    让一句话只跟真正出现它的那一小块挂钩，治'大块混进不相关词'的噪声。"""
    out = []
    units = []
    buf = []
    for ln in body.split("\n"):
        s = ln.strip()
        if not s:
            if buf:
                units.append("\n".join(buf)); buf = []
            continue
        if re.match(r"^([-*+]|\d+[.)])\s", s):
            if buf:
                units.append("\n".join(buf)); buf = []
            units.append(s)
        else:
            buf.append(s)
    if buf:
        units.append("\n".join(buf))
    for u in units:
        u = u.strip()
        if not u:
            continue
        if len(u) <= _CFG["block_max"]:
            out.append((head, u))
            continue
        parts = re.split(r"(?<=[。！？!?；;])", u)
        b = ""
        for pt in parts:
            if b and len(b) + len(pt) > _CFG["block_max"]:
                out.append((head, b)); b = pt
            else:
                b += pt
        if b.strip():
            out.append((head, b))
    return out


def _extract_hour(text):
    """从正文找 'HH:MM' 或 'HH点'，返回小时(0-23)，找不到返回 None。"""
    m = re.search(r"(\d{1,2})[:：](\d{2})", text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h
    m = re.search(r"(\d{1,2})\s*点", text)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h
    return None
def _split_blocks(text):
    blocks = []
    head = ""
    buf = []
    for ln in text.split("\n"):
        if re.match(r"^#{1,6}\s", ln.strip()):
            if buf and "".join(buf).strip():
                blocks.extend(_split_fine(head, "\n".join(buf)))
            head = ln.strip()
            buf = []
        else:
            buf.append(ln)
    if buf and "".join(buf).strip():
        blocks.extend(_split_fine(head, "\n".join(buf)))
    return blocks


def _guess_time(fname, fpath):
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", fname)
    if m:
        d = "%s-%s-%s" % m.groups()
    else:
        m2 = re.search(r"(\d{8})", fname)
        if m2:
            s2 = m2.group(1)
            d = "%s-%s-%s" % (s2[:4], s2[4:6], s2[6:8])
        else:
            d = datetime.date.fromtimestamp(os.path.getmtime(fpath)).isoformat()
    period = ""
    for pp in PERIOD_HOUR:
        if pp in fname:
            period = pp
            break
    hh = PERIOD_HOUR.get(period, 12)
    return {"date": d, "period": period or "日",
            "time": "%sT%02d:00:00" % (d, hh), "hour": hh}


def _tag(text):
    r = {"people": [], "topics": [], "things": []}
    for name, kws in PEOPLE.items():
        if any(k in text for k in kws):
            r["people"].append(name)
    for name, kws in TOPICS.items():
        if any(k in text for k in kws):
            r["topics"].append(name)
    for name, kws in THINGS.items():
        if any(k in text for k in kws):
            r["things"].append(name)
    return r


def _first_sent(body):
    for ln in body.split("\n"):
        t = ln.strip().lstrip("#-*> ").strip()
        if len(t) >= 6:
            return t[:46]
    return ""


def build_index(memdir, outpath=None):
    """扫描记忆目录，重建 index.json。返回 (条数, 输出路径)。"""
    import hashlib
    if not memdir or not os.path.isdir(memdir):
        raise RuntimeError("记忆目录不存在：" + str(memdir))
    if not outpath:
        outpath = os.path.join(_HERE, "index.json")
    entries = []
    for fname in sorted(os.listdir(memdir)):
        if not fname.lower().endswith((".md", ".txt")):
            continue
        fpath = os.path.join(memdir, fname)
        text = io.open(fpath, encoding="utf-8", errors="ignore").read()
        t = _guess_time(fname, fpath)
        for i, (head, body) in enumerate(_split_blocks(text)):
            hh = _extract_hour(body)
            if len(body.strip()) < 12:
                continue
            e = {
                "id": "%s#%d" % (os.path.splitext(fname)[0], i),
                "source": fname,
                "anchor": hashlib.sha1(body.strip()[:80].encode("utf-8")).hexdigest()[:12],
                "time": t["time"], "date": t["date"], "period": t["period"],
                "hour": hh if hh is not None else t["hour"],
                "head": head[:60], "summary": _first_sent(body),
                "text": re.sub(r"\s+", " ", body.strip())[:1500],
                "access_count": 0, "last_access": "",
            }
            e.update(_tag(head + "\n" + body))
            entries.append(e)
    io.open(outpath, "w", encoding="utf-8").write(json.dumps(entries, ensure_ascii=False, indent=1))
    # 索引变了：清缓存，下次查询重新读盘
    _IDX["data"] = None
    _IDX["vocab"] = None
    _IDX["df"] = None
    _IDX["path"] = outpath
    return len(entries), outpath


def _index_entries():
    return _load_index()


def fetch(source=None, keyword=None, limit=3):
    # ⑧先按折叠卡 id 取回
    if source and re.fullmatch(r's\d{10,}', str(source).strip()):
        t = fetch_settled(str(source).strip())
        if t:
            return t

    """取书：按出处文件名（source）或关键字，返回记忆原文。"""
    rows = []
    if source:
        sl = source.lower()
        for e in _index_entries():
            if sl in (e.get("source") or "").lower():
                rows.append(e)
    elif keyword:
        kl = keyword.lower()
        for e in _index_entries():
            blob = (_field_text(e, "head") + _field_text(e, "summary") + _field_text(e, "text")).lower()
            if kl in blob:
                rows.append(e)
    rows.sort(key=lambda e: e.get("time") or "")
    out = ["# 取原文（命中 %d 段）" % len(rows), ""]
    for e in rows[:limit]:
        out.append("## [%s] %s（%s）" % (rel_time(e.get("date", "")),
                                       (e.get("head") or "").lstrip("#").strip(),
                                       e.get("source") or "-"))
        out.append(e.get("text") or "")
        out.append("")
    return "\n".join(out)


def profile_card(who, limit=40):
    """按人聚卡：把索引里 people 命中该人的条目，按时间聚成一页。"""
    if not who:
        return ""
    wl = who.lower()
    rows = []
    for e in _index_entries():
        ppl = " ".join(str(x) for x in (e.get("people") or [])).lower()
        if wl in ppl:
            rows.append(e)
    if not rows:
        for e in _index_entries():
            blob = (_field_text(e, "head") + _field_text(e, "summary") + _field_text(e, "text")).lower()
            if wl in blob:
                rows.append(e)
    rows.sort(key=lambda e: e.get("time") or "")
    out = ["# 画像卡 · " + who, "", "共 %d 条（近 %d 条）" % (len(rows), min(len(rows), limit)), ""]
    for e in rows[-limit:]:
        head = (e.get("head") or "").lstrip("#").strip()
        summ = (e.get("summary") or e.get("text") or "").strip()[:60]
        out.append("- [%s] %s：%s（%s）" % (rel_time(e.get("date", "")), head, summ, e.get("source") or "-"))
    return "\n".join(out)


def open_loops(keyword=None, limit=20):
    """钓牵挂与欠账：扫索引里的待办/牵挂词。"""
    kws = [keyword] if keyword else ["牵挂", "欠账", "待办", "还欠", "没动笔", "未解决", "待续", "挂着", "等"]
    rows = []
    for e in _index_entries():
        blob = (_field_text(e, "head") + _field_text(e, "summary") + _field_text(e, "text"))
        hit = [k for k in kws if k and k in blob]
        if hit:
            rows.append((e, hit))
    rows.sort(key=lambda x: x[0].get("time") or "")
    out = ["# 牵挂与欠账（命中 %d 条）" % len(rows), ""]
    for e, hit in rows[-limit:]:
        head = (e.get("head") or "").lstrip("#").strip()
        summ = (e.get("summary") or e.get("text") or "").strip()[:70]
        out.append("- [%s] %s：%s（%s）" % (rel_time(e.get("date", "")), head, summ, e.get("source") or "-"))
    return "\n".join(out)


_MODEL = {"ready": False, "tok": None, "mdl": None, "device": "unavailable",
          "torch": None, "F": None, "err": ""}
_VEC_CACHE = {}


def _settle_dir():
    base = os.path.dirname(_IDX.get("path") or _find_index() or "") or "."
    d = os.path.join(base, "settled")
    os.makedirs(d, exist_ok=True)
    return d


def settle(text, card_id=None, keep=180):
    """⑧终局结算：把长输出折成一张卡片（id+摘要），正文挪到 settled/{id}.md。
    返回 (card_id, 卡片文本)。历史里只留卡片，需要时按 id 取回。"""
    text = text or ""
    if not card_id:
        card_id = "s" + datetime.datetime.now().strftime("%y%m%d%H%M%S")
    fn = os.path.join(_settle_dir(), card_id + ".md")
    io.open(fn, "w", encoding="utf-8").write(text)
    head = re.sub(r"\s+", " ", text.strip())[:keep]
    card = "[长输出已折叠·%d字·id=%s] %s…（需要时用 memfetch id=%s 取回）" % (len(text), card_id, head, card_id)
    return card_id, card


def fetch_settled(card_id):
    fn = os.path.join(_settle_dir(), (card_id or "").strip() + ".md")
    if os.path.exists(fn):
        return io.open(fn, encoding="utf-8").read()
    return ""


def _ensure_model():
    if _MODEL["ready"]:
        return
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    model_dir = find_model_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModel.from_pretrained(model_dir).to(device).eval()
    _MODEL.update({"ready": True, "tok": tok, "mdl": mdl, "device": device,
                   "torch": torch, "F": F})


def _embed(texts):
    """向量化；失败返回 None（调用方一律退回纯词频）。"""
    try:
        _ensure_model()
        tok, mdl, torch, F, device = _MODEL["tok"], _MODEL["mdl"], _MODEL["torch"], _MODEL["F"], _MODEL["device"]
        batch = tok(texts, padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**batch)
        vec = out.last_hidden_state[:, 0, :].float()
        vec = F.normalize(vec, dim=-1)
        return vec.cpu().tolist()
    except Exception as e:
        _MODEL["err"] = "%s: %s" % (type(e).__name__, e)
        return None


def _entry_vector(entry):
    """每条记忆的向量，进程内缓存（键=id）。"""
    key = entry.get("id") or entry.get("anchor") or ""
    v = _VEC_CACHE.get(key)
    if v is not None:
        return v
    vs = _embed([_field_text(entry, "summary") + " " + _field_text(entry, "text")[:300]])
    if not vs:
        return None
    _VEC_CACHE[key] = vs[0]
    return vs[0]


def _cos(a, b):
    try:
        return sum(x * y for x, y in zip(a, b))
    except Exception:
        return 0.0


def main():

    def reply(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    # 启动先试着把模型装上（装不上也不退出，recall 还能用）
    try:
        _ensure_model()
        reply({"id": 0, "fn": "ready", "ok": True, "device": _MODEL["device"], "index": _find_index()})
    except Exception as e:
        reply({"id": 0, "fn": "ready", "ok": True, "device": "unavailable",
               "error": "%s: %s" % (type(e).__name__, e), "index": _find_index()})

    # 索引缺失或为空：请 C# 侧后台触发 build（这里不擅自扫目录，路径由 C# 给）

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        fn = req.get("fn")
        rid = req.get("id", 0)
        if fn == "__shutdown__":
            break
        try:
            if fn == "ping":
                reply({"id": rid, "ok": True, "device": _MODEL["device"], "dim": 512})
            elif fn == "embed":
                texts = req.get("texts") or []
                vecs = _embed(texts) or []
                reply({"id": rid, "ok": True, "vectors": vecs})
            elif fn == "recall":
                _set_index(req.get("index"))
                q = req.get("query") or ""
                k = int(req.get("top_k") or 5)
                if req.get("half_life"):
                    _CFG["half_life"] = float(req.get("half_life"))
                hits = recall(q, top_k=k,
                              date_from=req.get("date_from") or None,
                              date_to=req.get("date_to") or None,
                              period=req.get("period") or None,
                              hour_from=req.get("hour_from"), hour_to=req.get("hour_to"),
                              use_graph=bool(req.get("use_graph", True)))
                reply({"id": rid, "ok": True, "bookmarks": to_bookmarks(hits), "count": len(hits)})
            elif fn == "build":
                if req.get("block_max"):
                    _CFG["block_max"] = int(req.get("block_max"))
                memdir = req.get("memdir") or ""
                outpath = req.get("out") or ""
                n, outp = build_index(memdir, outpath)
                _set_index(outp)
                reply({"id": rid, "ok": True, "count": n, "out": outp})
            elif fn == "fetch":
                _set_index(req.get("index"))
                txt = fetch(req.get("source"), req.get("keyword"), int(req.get("limit") or 3))
                reply({"id": rid, "ok": True, "text": txt})
            elif fn == "profile":
                txt = profile_card(req.get("who") or "", int(req.get("limit") or 40))
                reply({"id": rid, "ok": True, "text": txt})
            elif fn == "timeparse":
                df, dt, pd = parse_time_hint(req.get("text") or "")
                reply({"id": rid, "ok": True, "date_from": df, "date_to": dt, "period": pd})
            elif fn == "view":
                _set_index(req.get("index"))
                n, outp = render_view(req.get("out"))
                reply({"id": rid, "ok": True, "count": n, "out": outp})
            elif fn == "graph":
                _set_index(req.get("index"))
                n, ed = build_graph()
                reply({"id": rid, "ok": True, "nodes": n, "edges": ed})
            elif fn == "maintain":
                _set_index(req.get("index"))
                r = maintain_index(req.get("memdir"), allow_soft=bool(req.get("allow_soft")))
                reply({"id": rid, "ok": True, **r})
            elif fn == "settle":
                _set_index(req.get("index"))
                cid, card = settle(req.get("text") or "", req.get("card_id"), int(req.get("keep") or 180))
                reply({"id": rid, "ok": True, "card_id": cid, "card": card})
            elif fn == "loops":
                txt = open_loops(req.get("keyword"), int(req.get("limit") or 20))
                reply({"id": rid, "ok": True, "text": txt})
            else:
                reply({"id": rid, "ok": False, "error": "unknown fn: %s" % fn})
        except Exception as e:
            reply({"id": rid, "ok": False, "error": "%s: %s" % (type(e).__name__, e)})


if __name__ == "__main__":
    main()
