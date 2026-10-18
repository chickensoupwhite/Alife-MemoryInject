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
# 多角色共用同一个侧车进程：按"索引路径"分桶缓存，切角色时不必反复冷读。
_CACHE = {}         # path -> {"data":..., "vocab":..., "df":...}
_CACHE_ORDER = []   # LRU 顺序（最近用的排后）
_CACHE_MAX = 3      # 最多缓存几份完整索引（三四个角色够用，不无限吃内存）


def _cache_put(path, key, val):
    if not path:
        return
    c = _CACHE.setdefault(path, {"data": None, "vocab": None, "df": None})
    c[key] = val
    if path in _CACHE_ORDER:
        _CACHE_ORDER.remove(path)
    _CACHE_ORDER.append(path)
    while len(_CACHE_ORDER) > _CACHE_MAX:
        _CACHE.pop(_CACHE_ORDER.pop(0), None)


def _find_index():
    """找 index.json：先看脚本同目录，再看上级目录。"""
    for c in (os.path.join(_HERE, "index.json"),
              os.path.join(os.path.dirname(_HERE), "index.json")):
        if os.path.isfile(c):
            return c
    return None


def _set_index(path):
    """切换当前使用的索引文件（多角色各自一份，不打架）。
    命中分桶缓存就直接用热的，避免多角色来回切时反复冷读。
    ★显式路径即使文件暂时不在也要记下来：绝不回退到别人的索引。"""
    if path and _IDX.get("path") != path:
        _GRAPH["built"] = False      # ★换索引=换图：不清就会拿上一个角色的图去扩词
        _IDX["path"] = path
        c = _CACHE.get(path)
        if c is not None:
            _IDX["data"] = c.get("data")
            _IDX["vocab"] = c.get("vocab")
            _IDX["df"] = c.get("df")
            if path in _CACHE_ORDER:
                _CACHE_ORDER.remove(path)
            _CACHE_ORDER.append(path)
        else:
            _IDX["data"] = None
            _IDX["vocab"] = None
            _IDX["df"] = None


def _load_index():
    """加载索引（带缓存）。显式路径给了但文件不在 → 返回空表，绝不回退到别人的索引。"""
    p = _IDX["path"]
    if p is None:
        p = _find_index()
        _IDX["path"] = p
    # ★数据库优先：建过库就以库为准（JSON 只用于首次迁移/导出）
    try:
        if p and _DB.get("ok") and _IDX["data"] is None:
            rows = db_load_all(p)
            if rows is not None:
                _IDX["data"] = rows
                return rows
    except Exception:
        pass
    if _IDX["data"] is None:
        if p and os.path.isfile(p):
            try:
                with open(p, encoding="utf-8") as f:
                    _IDX["data"] = json.load(f)
            except Exception:
                _IDX["data"] = []
            _cache_put(_IDX["path"], "data", _IDX["data"])
        else:
            return []   # 显式路径缺失：空表，不缓存，等 build 出来再读
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
        _cache_put(_IDX["path"], "vocab", _IDX["vocab"])
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
    _cache_put(_IDX["path"], "df", df)
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
           hour_from=None, hour_to=None, use_graph=True, skip_recent_days=0):
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
    if int(skip_recent_days or 0) > 0:
        _cut = (datetime.date.today() - datetime.timedelta(days=int(skip_recent_days))).isoformat()
        data = [e for e in data if not (e.get("date") or "") or str(e.get("date")) < _cut]
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
                # ★以前只匹配 people/topics/things，实体节点等于白建；正文/标题/摘要也一起匹配
                etags = " ".join([_field_text(e, "people"), _field_text(e, "topics"),
                                  _field_text(e, "things"), _field_text(e, "head"),
                                  _field_text(e, "summary"), _field_text(e, "text")]).lower()
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
        # ★大库保护：关键词路已按分数降序，只对前 N 条做语义精排。
        #   之前对全部命中逐条 embed，大库(3万条)会拖到 1s+，超出同步等待窗口。
        SEM_TOP = 80
        cand = scored[:SEM_TOP]
        vecs = [None] * len(cand)
        to_embed, to_idx = [], []
        for i, (_, _, _, e) in enumerate(cand):
            key = e.get("id") or e.get("anchor") or ""
            v = _VEC_CACHE.get(key)
            if v is not None:
                vecs[i] = v
            else:
                to_embed.append(_field_text(e, "summary") + " " + _field_text(e, "text")[:300])
                to_idx.append(i)
        if to_embed:
            vs = _embed(to_embed) or []
            for j, i in enumerate(to_idx):
                if j < len(vs):
                    vecs[i] = vs[j]
                    e = cand[i][3]
                    _VEC_CACHE[e.get("id") or e.get("anchor") or ""] = vs[j]
        sims = []
        for i, (_, _, _, e) in enumerate(cand):
            if vecs[i]:
                sims.append((_cos(qv[0], vecs[i]), e))
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
    if _DB.get("ok"):
        _LIFE["dirty"] = False
        return          # 数据库模式：生命周期已直接 UPDATE 到库，不再回写 JSON
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
        # ⑤ 正文实体（规则抽取，不上模型）：并进同一张图，节点比"只有标签"多一层
        ent_text = "%s %s %s" % (e.get("head") or "", e.get("summary") or "", str(e.get("text") or "")[:400])
        for x in extract_entities(ent_text)[:6]:
            ns.append(x)
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


def links(query, top=6, max_seed=3):
    """图谱邻居：某个节点跟谁常一起出现。给 AI 顺着关系找线索用（不是只做隐式扩词）。"""
    if not _GRAPH["built"]:
        build_graph()
    q = (query or "").strip().lower()
    seeds = []
    if q:
        for n in _GRAPH["nodes"]:
            nl = n.lower()
            if nl in q or (len(nl) >= 2 and q in nl):
                seeds.append(n)
    seeds.sort(key=lambda n: -_GRAPH["nodes"].get(n, 0))
    seeds = seeds[:max(1, int(max_seed))]
    items = []
    for s in seeds:
        items.append({
            "node": s,
            "count": _GRAPH["nodes"].get(s, 0),
            "neighbors": [{"name": n, "w": w} for n, w in _neighbors(s, top)],
        })
    if not items:
        return {"seeds": [], "items": [], "text": ""}
    lines = []
    for it in items:
        nb = "、".join("%s(%d)" % (x["name"], x["w"]) for x in it["neighbors"]) or "（没有邻居）"
        lines.append("- %s：出现 %d 次；常一起出现 → %s" % (it["node"], it["count"], nb))
    return {"seeds": seeds, "items": items, "text": "\n".join(lines)}


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


def build_index(memdir, outpath=None, extra_dirs=None, exclude=None):
    """扫描记忆目录重建 index.json。
    extra_dirs：额外目录（我们自己的事实层），其 source 统一加 facts/ 前缀；
    exclude：要跳过的相对路径（已压缩的官方存档，只是不建索引，文件原样保留）。"""
    import hashlib
    if not memdir or not os.path.isdir(memdir):
        raise RuntimeError("记忆目录不存在：" + str(memdir))
    if not outpath:
        outpath = os.path.join(_HERE, "index.json")
    # 读旧索引：按 anchor 保留生命周期字段（access_count/last_access/retired），
    # 让"自动整理"重建时不丢"访问强化/衰减/软删"状态。
    old = {}
    if os.path.exists(outpath):
        try:
            for _e in json.load(io.open(outpath, encoding="utf-8")):
                _a = _e.get("anchor")
                if _a:
                    old[_a] = _e
        except Exception:
            old = {}
    entries = []
    skip_dirs = {"settled", "backup", "备份", "archive", "MemoryFacts"}
    ex = set(str(x).replace("\\", "/") for x in (exclude or []))
    jobs = [(memdir, "")]
    for d in (extra_dirs or []):
        if d and os.path.isdir(d):
            jobs.append((d, "facts/"))
    allfiles = []
    for base, prefix in jobs:
        for root, dirs, _files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for fn in _files:
                if fn.lower().endswith((".md", ".txt")) and ".bak" not in fn.lower():
                    allfiles.append((os.path.join(root, fn), base, prefix))
    for fpath, base, prefix in sorted(allfiles, key=lambda x: x[0]):
        fname = os.path.basename(fpath)
        rel = prefix + os.path.relpath(fpath, base).replace("\\", "/")
        if prefix == "" and _is_excluded(rel, ex):
            continue        # 已压缩/手动排除：只跳过索引，官方文件一个字节都不动
        text = io.open(fpath, encoding="utf-8", errors="ignore").read()
        t = _guess_time(fname, fpath)
        for i, (head, body) in enumerate(_split_blocks(text)):
            hh = _extract_hour(body)
            if len(body.strip()) < 12:
                continue
            e = {
                "id": "%s#%d" % (os.path.splitext(rel)[0], i),
                "source": rel,
                "anchor": hashlib.sha1(body.strip()[:80].encode("utf-8")).hexdigest()[:12],
                "time": t["time"], "date": t["date"], "period": t["period"],
                "hour": hh if hh is not None else t["hour"],
                "head": head[:60], "summary": _first_sent(body),
                "text": re.sub(r"\s+", " ", body.strip())[:1500],
                "access_count": 0, "last_access": "",
            }
            e.update(_tag(head + "\n" + body))
            # 保留生命周期：同一 anchor（内容指纹）的旧条目，沿用其访问计数/软删状态
            _prev = old.get(e["anchor"])
            if _prev:
                e["access_count"] = _prev.get("access_count", 0)
                e["last_access"] = _prev.get("last_access", "")
                if _prev.get("retired"):
                    e["retired"] = _prev["retired"]
                    e["retired_at"] = _prev.get("retired_at")
            entries.append(e)
    io.open(outpath, "w", encoding="utf-8").write(json.dumps(entries, ensure_ascii=False, indent=1))
    # 索引变了：清缓存，下次查询重新读盘
    _GRAPH["built"] = False          # ★索引重建后图谱也必须重建，否则 links/graph_expand 用旧图
    _GRAPH["nodes"] = {}
    _GRAPH["edges"] = {}
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


def bytime(date_from=None, date_to=None, period=None, hour_from=None, hour_to=None, top_k=10):
    """按时间范围列记忆（不要求关键词）：日/时段/小时都能下钻，按时间倒序返回。"""
    data = [e for e in _load_index() if not e.get("retired")]

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
    data.sort(key=lambda e: str(e.get("time") or ""), reverse=True)
    return data[:max(1, int(top_k or 10))]


def bywhen(text, keyword=None, top_k=8, use_graph=True):
    """一句中文时间词 -> 那段时间的记忆。「上周三」「三天前」「昨晚」都行；给关键词就按相关度排。"""
    df, dt, pd = parse_time_hint(text or "")
    if keyword:
        hits = recall(keyword, top_k=top_k, date_from=df, date_to=dt, period=pd, use_graph=use_graph)
    else:
        hits = bytime(date_from=df, date_to=dt, period=pd, top_k=top_k)
    return {
        "bookmarks": to_bookmarks(hits),
        "count": len(hits),
        "date_from": df or "",
        "date_to": dt or "",
        "period": pd or "",
        "text": "%s ~ %s%s 共 %d 条" % (df or "?", dt or "?", (" " + pd) if pd else "", len(hits)),
    }


def index_stats():
    """索引体量与时间跨度，供 memstatus 用。"""
    data = [e for e in _load_index() if not e.get("retired")]
    dates = sorted({str(e.get("date") or "") for e in data if e.get("date")})
    return {"count": len(data), "first": dates[0] if dates else "", "last": dates[-1] if dates else ""}


# ============================================================
# ① 自动压缩 / ② 审计 / ③ 写入合并 / ⑤ 实体图谱 / ⑥ 浏览
# ============================================================

def _tokens(text):
    """粗粒度 token（中文按 2-gram、英文按词），只用于相似度比较。"""
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", str(text or ""))
    out = set()
    for w in s.split():
        if len(w) <= 2:
            out.add(w)
        else:
            for i in range(len(w) - 1):
                out.add(w[i:i + 2])
    return out


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return (inter / float(uni)) if uni else 0.0


def _sim(a, b):
    """事实相似度：Jaccard 与"包含度"取大。
    短句（如「血压偏高」vs「血压还是偏高些」）Jaccard 会被稀释，包含度能抓住。"""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    j = inter / float(len(a | b))
    c = inter / float(min(len(a), len(b)))
    return max(j, c * 0.9)


# ---------- ① 自动压缩 ----------

def compact_scan(memdir, min_entries=30, min_age_days=7, max_chars=20000, top_k=8, exclude=None):
    """找可压缩的流水文件：条目多、够旧。返回候选（含要喂给模型的正文）。
    ★不动原文：这里只给候选，C# 侧提炼、归档、重建索引。"""
    groups = {}
    for e in _load_index():
        if e.get("retired"):
            continue
        src = str(e.get("source") or "")
        if not src:
            continue
        groups.setdefault(src, []).append(e)
    today = datetime.date.today()
    ex = set(str(x).replace("\\", "/") for x in (exclude or []))
    cands = []
    for src, items in groups.items():
        if _is_excluded(src, ex) or src.startswith("facts/"):
            continue        # 已压缩过 / 手动排除 / 我们自己的事实层，都不再压
        if len(items) < int(min_entries):
            continue
        dates = sorted(str(x.get("date") or "") for x in items if x.get("date"))
        age = 0
        if dates:
            try:
                age = (today - datetime.date.fromisoformat(dates[-1])).days
            except Exception:
                age = 0
        if age < int(min_age_days):
            continue
        items.sort(key=lambda x: str(x.get("time") or ""))
        parts, size = [], 0
        for x in items:
            who = "".join(x.get("people") or [])
            topic = "".join(x.get("topics") or [])
            summary = str(x.get("summary") or x.get("head") or "").strip()
            line = " | ".join([str(x.get("date") or ""), who, topic, summary])
            size += len(line)
            if size > int(max_chars):
                break
            parts.append(line)
        cands.append({
            "source": src,
            "entries": len(items),
            "chars": size,
            "date_from": dates[0] if dates else "",
            "date_to": dates[-1] if dates else "",
            "text": "\n".join(parts),
        })
    cands.sort(key=lambda c: (-c["entries"], c["date_from"]))
    return cands[:max(1, int(top_k))]


# ---------- ② 审计 ----------

def _fact_files(memdir):
    out = []
    try:
        for name in os.listdir(memdir):
            if name.startswith("事实_") and name.endswith(".md"):
                out.append(os.path.join(memdir, name))
    except Exception:
        pass
    out.sort()
    return out


def _parse_fact(line):
    """解析事实卡行：- 日期 | 涉及谁 | 属哪件事 | 一句话事实"""
    s = str(line or "").strip()
    if s.startswith("-"):
        s = s[1:].strip()
    if "|" not in s:
        return None
    parts = [p.strip() for p in s.split("|")]
    if len(parts) < 4:
        return None
    return {"date": parts[0], "who": parts[1], "topic": parts[2], "fact": "|".join(parts[3:]).strip()}


_NEG = ("没", "未", "不", "取消", "放弃", "作废")


def audit_scan(memdir, limit=30, dup=0.85, near=0.5, max_facts=400):
    """扫事实卡：重复 / 可能重复 / 可能矛盾，只出报告，不自动改。"""
    facts = []
    for path in _fact_files(memdir):
        try:
            with io.open(path, encoding="utf-8") as fh:
                for ln in fh:
                    f = _parse_fact(ln)
                    if f:
                        f["file"] = os.path.basename(path)
                        f["line"] = ln.rstrip()
                        facts.append(f)
        except Exception:
            continue
    # ★只比最近的一批：事实卡上千条时 O(n^2) 会拖死侧车
    if len(facts) > int(max_facts):
        facts = facts[-int(max_facts):]
    out = []
    n = len(facts)
    cap = max(1, int(limit)) * 3
    for i in range(n):
        if len(out) >= cap:
            break
        for j in range(i + 1, n):
            a, b = facts[i], facts[j]
            same_key = (a["who"] == b["who"] and a["topic"] == b["topic"])
            sim = _sim(_tokens(a["fact"]), _tokens(b["fact"]))
            kind = None
            if sim >= float(dup):
                kind = "重复"
            elif same_key and sim >= float(near):
                kind = "可能重复"
            elif same_key:
                na = any(w in a["fact"] for w in _NEG)
                nb = any(w in b["fact"] for w in _NEG)
                if na != nb:
                    kind = "可能矛盾"
            if kind:
                out.append({
                    "kind": kind,
                    "score": round(sim, 3),
                    "a": a["line"], "b": b["line"],
                    "file_a": a["file"], "file_b": b["file"],
                })
    out.sort(key=lambda x: (0 if x["kind"] == "重复" else 1, -x["score"]))
    return out[:max(1, int(limit))]


# ---------- ③ 写入合并 ----------

def fact_merge(new_lines, existing_text="", threshold=0.5):
    """新事实入库前合并：同主体同类别里，重复的丢、相近的并、其余追加。"""
    existing_text = existing_text or ""
    out_lines = existing_text.splitlines()
    existing = []
    for ln in out_lines:
        f = _parse_fact(ln)
        if f:
            f["line"] = ln.rstrip()
            existing.append(f)
    pos = {}
    for i, ln in enumerate(out_lines):
        pos.setdefault(ln.rstrip(), i)

    added, merged, skipped = [], [], []
    for raw in (new_lines or []):
        raw = str(raw or "").strip()
        if not raw:
            continue
        f = _parse_fact(raw)
        if not f:
            out_lines.append("- " + raw)
            added.append(raw)
            continue
        best_i, best_s = -1, 0.0
        for i, old in enumerate(existing):
            if old["who"] != f["who"] or old["topic"] != f["topic"]:
                continue
            s = _sim(_tokens(old["fact"]), _tokens(f["fact"]))
            if s > best_s:
                best_s, best_i = s, i
        if best_i >= 0 and best_s >= 0.85:
            old = existing[best_i]
            # 近似重复但新句子明显更详细：只刷新日期会丢掉新增信息 → 按"合并"保留更长的描述
            if len(f["fact"]) >= len(old["fact"]) + 6:
                keep = f["fact"]
                date = max(f["date"], old["date"]) if (f["date"] and old["date"]) else (f["date"] or old["date"])
                line = "- %s | %s | %s | %s" % (date, old["who"], old["topic"], keep)
                k = pos.get(old["line"])
                if k is not None:
                    out_lines[k] = line
                old["fact"] = keep
                old["line"] = line
                old["date"] = date
                merged.append(line)
                continue
            # 同一事实再次出现：不重复入库，但把日期刷新到最新（旧事实不该永远停在老日期）
            if f["date"] and f["date"] > old["date"]:
                line = "- %s | %s | %s | %s" % (f["date"], old["who"], old["topic"], old["fact"])
                k = pos.get(old["line"])
                if k is not None:
                    out_lines[k] = line
                old["line"] = line
                old["date"] = f["date"]
            skipped.append(raw)
            continue
        if best_i >= 0 and best_s >= float(threshold):
            old = existing[best_i]
            keep = f["fact"] if len(f["fact"]) > len(old["fact"]) else old["fact"]
            date = max(f["date"], old["date"]) if (f["date"] and old["date"]) else (f["date"] or old["date"])
            line = "- %s | %s | %s | %s" % (date, old["who"], old["topic"], keep)
            k = pos.get(old["line"])
            if k is not None:
                out_lines[k] = line
            old["fact"] = keep
            old["line"] = line
            merged.append(line)
            continue
        line = "- %s | %s | %s | %s" % (f["date"], f["who"], f["topic"], f["fact"])
        out_lines.append(line)
        added.append(line)
    return {
        "text": "\n".join(out_lines),
        "added": added,
        "merged": merged,
        "skipped": skipped,
        "summary": "新增 %d，合并 %d，跳过重复 %d" % (len(added), len(merged), len(skipped)),
    }


# ---------- ⑤ 实体抽取（规则，不上模型） ----------

_ENTITY_PATTERNS = [
    re.compile(r"《([^》]{1,20})》"),
    re.compile(r"「([^」]{1,20})」"),
    re.compile(r"『([^』]{1,20})』"),
    re.compile(r"([\u4e00-\u9fff]{2,6}(?:插件|项目|模型|客户端|服务|工具|游戏|软件))"),
]

_ENTITY_STOP = set(["什么", "这个", "那个", "我们", "他们", "自己", "现在", "今天", "昨天", "明天"])


def extract_entities(text):
    """从正文抽实体（书名号/引号/专名后缀/拉丁词），不用模型，避免吃显存。"""
    t = str(text or "")
    out = []
    for rx in _ENTITY_PATTERNS:
        for m in rx.finditer(t):
            out.append((m.group(1) if m.groups() else m.group(0)).strip())
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_.\-]{2,20}", t):
        out.append(m.group(0))
    seen, res = set(), []
    for x in out:
        x = x.strip(" 　,，。.;；:：")
        if not x or len(x) < 2 or x in _ENTITY_STOP or x in seen:
            continue
        seen.add(x)
        res.append(x)
    return res


# ---------- ⑥ 内嵌浏览 ----------

def browse(query=None, who=None, topic=None, thing=None, date_from=None, date_to=None, offset=0, limit=50):
    """给内嵌面板用：按关键词/人/事/物/时间翻条目，返回结构化列表 + 可直接显示的文本。"""
    data = [e for e in _load_index() if not e.get("retired")]
    q = (query or "").strip().lower()

    def _hay(e):
        # 只拼相关字段：比 json.dumps 整条便宜得多（3 万条时差别很明显）
        return " ".join([
            " ".join(e.get("people") or []),
            " ".join(e.get("topics") or []),
            " ".join(e.get("things") or []),
            str(e.get("head") or ""),
            str(e.get("summary") or ""),
            str(e.get("text") or ""),
        ]).lower()

    def _ok(e):
        if q and q not in _hay(e):
            return False
        if who and who not in (e.get("people") or []):
            return False
        if topic and topic not in (e.get("topics") or []):
            return False
        if thing and thing not in (e.get("things") or []):
            return False
        d = str(e.get("date") or "")
        if date_from and d and d < date_from:
            return False
        if date_to and d and d > date_to:
            return False
        return True

    hits = [e for e in data if _ok(e)]
    hits.sort(key=lambda e: str(e.get("time") or ""), reverse=True)
    total = len(hits)
    off = max(0, int(offset or 0))
    lim = max(1, min(200, int(limit or 50)))
    items = []
    for e in hits[off:off + lim]:
        items.append({
            "id": e.get("id") or "",
            "date": e.get("date") or "",
            "period": e.get("period") or "",
            "head": (e.get("head") or "").lstrip("#").strip(),
            "summary": e.get("summary") or "",
            "source": e.get("source") or "",
            "people": e.get("people") or [],
            "topics": e.get("topics") or [],
            "things": e.get("things") or [],
        })
    lines = []
    for it in items:
        label = "、".join(list(it["topics"]) + list(it["things"])) or it["head"] or "记忆"
        lines.append("- [%s] %s：%s（出处 %s）" % (it["date"], label, it["summary"], it["source"]))
    return {"total": total, "offset": off, "items": items, "text": "\n".join(lines)}


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


# ============================================================
# 数据库层（SQLite）：增量建库 + 按需查询
#   - 条目主键用 id（source#块号），anchor 只做去重/水位参考
#   - files 表按 mtime 做增量：只重扫变过的文件
#   - vec 表缓存条目向量（float32 blob），避免每次重算
#   - nodes/edges 表存互链（标签 + 正文实体共现）
#   - JSON 仅用于首次迁移与 memview 导出；查询不走 JSON
# ============================================================

import sqlite3
import struct
import hashlib

_DB = {"ok": False, "path": "", "conn": None}

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries(
  id TEXT PRIMARY KEY, anchor TEXT, source TEXT,
  time TEXT, date TEXT, period TEXT, hour INTEGER,
  head TEXT, summary TEXT, body TEXT, hay TEXT,
  people TEXT, topics TEXT, things TEXT,
  access_count INTEGER DEFAULT 0, last_access TEXT,
  retired INTEGER DEFAULT 0, retired_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_entries_time ON entries(time);
CREATE INDEX IF NOT EXISTS ix_entries_source ON entries(source);
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, mtime REAL);
CREATE TABLE IF NOT EXISTS vec(id TEXT PRIMARY KEY, data BLOB);
CREATE TABLE IF NOT EXISTS nodes(name TEXT PRIMARY KEY, n INTEGER);
CREATE TABLE IF NOT EXISTS edges(a TEXT, b TEXT, w INTEGER, type TEXT, PRIMARY KEY(a, b));
"""


def _is_excluded(rel, ex):
    """排除判定：支持精确路径与目录前缀（以 / 结尾）。"""
    r = (rel or "").replace("\\", "/")
    if not r:
        return False
    for e in (ex or []):
        e2 = str(e).replace("\\", "/")
        if e2.endswith("/"):
            if r.startswith(e2):
                return True
        elif r == e2:
            return True
    return False


def _db_open(db_path):
    try:
        if not db_path:
            return None
        c = sqlite3.connect(db_path, timeout=15)
        c.row_factory = sqlite3.Row
        c.executescript(_DB_SCHEMA)
        upgraded = False
        try:
            c.execute("ALTER TABLE edges ADD COLUMN type TEXT")
            upgraded = True
        except Exception:
            pass
        c.commit()
        if upgraded:
            try:
                _db_rebuild_links(c)   # 旧库升级：重算一遍带类型的边
                c.commit()
            except Exception:
                pass
        _DB.update({"ok": True, "path": db_path, "conn": c})
        return c
    except Exception:
        _DB.update({"ok": False, "path": db_path, "conn": None})
        return None


def _db_for(outpath):
    """按索引输出路径定位同目录的 memory.db（每角色一份）。"""
    dbp = os.path.join(os.path.dirname(outpath or ""), "memory.db") if outpath else ""
    if not dbp:
        return None
    if _DB["ok"] and _DB["path"] == dbp and _DB["conn"] is not None:
        return _DB["conn"]
    return _db_open(dbp)


def _row_entry(c, r):
    """SQL 行 -> 条目 dict（tags 还原成列表，兼容旧算法）。"""
    d = dict(r)
    d["text"] = d.pop("body", "")
    for k in ("people", "topics", "things"):
        v = d.get(k) or ""
        d[k] = [x for x in v.split("|") if x]
    d["retired"] = bool(d.get("retired"))
    return d


def _entry_row(e, hay=None):
    tags = " ".join((e.get("people") or []) + (e.get("topics") or []) + (e.get("things") or []))
    if hay is None:
        hay = " ".join([str(e.get("head") or ""), str(e.get("summary") or ""),
                        str(e.get("text") or ""), tags]).lower()
    return (
        e.get("id") or "", e.get("anchor") or "", e.get("source") or "",
        e.get("time") or "", e.get("date") or "", e.get("period") or "",
        e.get("hour"), (e.get("head") or "")[:60], e.get("summary") or "",
        e.get("text") or "", hay,
        "|".join(e.get("people") or []), "|".join(e.get("topics") or []),
        "|".join(e.get("things") or []),
        int(e.get("access_count") or 0), e.get("last_access") or "",
        1 if e.get("retired") else 0, e.get("retired_at") or "",
    )


def _db_upsert(c, e):
    try:
        c.execute(
            "INSERT OR REPLACE INTO entries(id,anchor,source,time,date,period,hour,head,summary,body,hay,"
            "people,topics,things,access_count,last_access,retired,retired_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _entry_row(e))
        return True
    except Exception:
        return False


def _db_import_json(c, json_path):
    """首次迁移：把旧的 index.json 导入库（只导一次）。"""
    try:
        if not json_path or not os.path.isfile(json_path):
            return 0
        if c.execute("SELECT COUNT(*) FROM entries").fetchone()[0] > 0:
            return 0
        data = json.load(io.open(json_path, encoding="utf-8")) or []
    except Exception:
        return 0
    n = 0
    for e in data:
        if _db_upsert(c, e):
            n += 1
    c.commit()
    if n:
        _db_rebuild_links(c)   # 迁移后补建互链，否则 links 会是空的
        c.commit()
    return n


def _cat(n, people, topics, things):
    if n in people:
        return "人"
    if n in topics:
        return "事"
    if n in things:
        return "物"
    return "实体"


def _edge_type(a, b, people, topics, things, ents):
    """互链类型：同人/同事/同物、人-事/人-物/事-物、实体、共现（可多类型，首项为主）。"""
    ca, cb = _cat(a, people, topics, things), _cat(b, people, topics, things)
    ts = []
    if ca == cb == "人":
        ts.append("同人")
    elif ca == cb == "事":
        ts.append("同事")
    elif ca == cb == "物":
        ts.append("同物")
    elif {ca, cb} == {"人", "事"}:
        ts.append("人-事")
    elif {ca, cb} == {"人", "物"}:
        ts.append("人-物")
    elif {ca, cb} == {"事", "物"}:
        ts.append("事-物")
    if "实体" in (ca, cb) and "实体" not in ts:
        ts.append("实体")
    if not ts:
        ts.append("共现")
    return "|".join(ts)


def _db_rebuild_links(c):
    """重建互链（标签 + 正文实体，共现边）。建库时跑一次，成本可控。"""
    try:
        c.execute("DELETE FROM nodes")
        c.execute("DELETE FROM edges")
        nodes, edges = {}, {}
        for r in c.execute("SELECT id,head,summary,body,people,topics,things FROM entries"):
            d = _row_entry(c, r)
            ns = []
            for k in ("people", "topics", "things"):
                ns.extend(d.get(k) or [])
            ent_text = "%s %s %s" % (d.get("head") or "", d.get("summary") or "", (d.get("text") or "")[:400])
            for x in extract_entities(ent_text)[:6]:
                ns.append(x)
            ns = list(dict.fromkeys([n for n in ns if n]))
            for n in ns:
                nodes[n] = nodes.get(n, 0) + 1
            people = d.get("people") or []
            topics = d.get("topics") or []
            things = d.get("things") or []
            ents = extract_entities(ent_text)
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    a, b = sorted((ns[i], ns[j]))
                    cur = edges.get((a, b))
                    if cur is None:
                        edges[(a, b)] = {"w": 1, "type": _edge_type(a, b, people, topics, things, ents)}
                    else:
                        cur["w"] += 1
                        if "实体" in cur["type"] and a not in ents and b not in ents:
                            pass
        c.executemany("INSERT OR REPLACE INTO nodes(name,n) VALUES(?,?)", list(nodes.items()))
        c.executemany("INSERT OR REPLACE INTO edges(a,b,w,type) VALUES(?,?,?,?)",
                      [(a, b, v["w"], v.get("type") or "共现") for (a, b), v in edges.items()])
        c.commit()
    except Exception:
        pass


def _db_ready(req):
    """取连接；库为空而旧 index.json 在就先迁移（兼容只给 index 的旧调用）。"""
    idx = req.get("index") or ""
    c = _db_for(idx)
    if c is None:
        return None
    try:
        if c.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0:
            _db_import_json(c, idx)
    except Exception:
        pass
    return c


def db_build(memdir, outpath=None, extra_dirs=None, exclude=None):
    """增量建库：只重扫 mtime 变过的文件；返回条目总数。"""
    c = _db_for(outpath)
    if c is None:
        return None
    _db_import_json(c, outpath)
    skip_dirs = {"settled", "backup", "备份", "archive", "MemoryFacts"}
    ex = set(str(x).replace("\\", "/") for x in (exclude or []))
    jobs = [(memdir, "")]
    for d in (extra_dirs or []):
        if d and os.path.isdir(d):
            jobs.append((d, "facts/"))
    seen = set()
    changed = False
    for base, prefix in jobs:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for fn in files:
                if not fn.lower().endswith((".md", ".txt")) or ".bak" in fn.lower():
                    continue
                fpath = os.path.join(root, fn)
                rel = prefix + os.path.relpath(fpath, base).replace("\\", "/")
                if prefix == "" and _is_excluded(rel, ex):
                    continue
                seen.add(rel)
                try:
                    mt = os.path.getmtime(fpath)
                except Exception:
                    continue
                row = c.execute("SELECT mtime FROM files WHERE path=?", (rel,)).fetchone()
                if row and abs(float(row["mtime"]) - mt) < 1e-6:
                    continue
                try:
                    text = io.open(fpath, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                t = _guess_time(fn, fpath)
                c.execute("DELETE FROM entries WHERE source=?", (rel,))
                for i, (head, body) in enumerate(_split_blocks(text)):
                    if len(body.strip()) < 12:
                        continue
                    hh = _extract_hour(body)
                    ent = {
                        "id": "%s#%d" % (os.path.splitext(rel)[0], i),
                        "anchor": hashlib.sha1(body.strip()[:80].encode("utf-8")).hexdigest()[:12],
                        "source": rel, "time": t["time"], "date": t["date"], "period": t["period"],
                        "hour": hh if hh is not None else t["hour"],
                        "head": head[:60], "summary": _first_sent(body), "text": body.strip(),
                    }
                    tags = _tag(body)
                    ent["people"] = tags.get("people") or []
                    ent["topics"] = tags.get("topics") or []
                    ent["things"] = tags.get("things") or []
                    _db_upsert(c, ent)
                c.execute("INSERT OR REPLACE INTO files(path,mtime) VALUES(?,?)", (rel, mt))
                changed = True
    # 文件已消失：清掉它的条目
    for r in c.execute("SELECT path FROM files").fetchall():
        p = r["path"]
        if p not in seen:
            c.execute("DELETE FROM entries WHERE source=?", (p,))
            c.execute("DELETE FROM files WHERE path=?", (p,))
            changed = True
    if changed:
        _db_rebuild_links(c)
        try:
            c.execute("DELETE FROM vec")
        except Exception:
            pass
        c.commit()
    n = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    return n


def _db_where_time(req, alias=""):
    where, args = [], []
    df = req.get("date_from") or None
    dt = req.get("date_to") or None
    pd = req.get("period") or None
    hf, ht = req.get("hour_from"), req.get("hour_to")
    if df:
        where.append("date >= ?"); args.append(df)
    if dt:
        where.append("date <= ?"); args.append(dt)
    if pd:
        where.append("period LIKE ?"); args.append("%" + pd + "%")
    if hf is not None:
        where.append("hour IS NOT NULL AND hour >= ?"); args.append(int(hf))
    if ht is not None:
        where.append("hour IS NOT NULL AND hour <= ?"); args.append(int(ht))
    where.append("retired = 0")
    return where, args


def _db_score(entry, terms):
    """与 JSON 版 score_entry 同规则（标签3 > 标题3 > 摘要2 > 正文1，按词频权重）。"""
    tags = " ".join((entry.get("people") or []) + (entry.get("topics") or []) + (entry.get("things") or [])).lower()
    head = (entry.get("head") or "").lower()
    summary = (entry.get("summary") or "").lower()
    text = (entry.get("text") or "").lower()
    total = 0.0
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


def _db_boost(entry):
    """强化 + 衰减（与 _entry_weight 同规则，但作用在数据库行上）。"""
    ac = min(int(entry.get("access_count") or 0), 5)
    reinforce = 1.0 + 0.05 * ac
    base = entry.get("last_access") or entry.get("date") or ""
    decay = 1.0
    try:
        if base:
            age = (datetime.date.today() - datetime.date.fromisoformat(str(base)[:10])).days
            decay = 0.5 ** (max(0, age) / float(_CFG.get("half_life", 30)))
    except Exception:
        decay = 1.0
    return max(0.6, min(1.25, reinforce * max(0.05, decay) + 0.25 * min(1.0, decay)))


def _db_vec_get(c, eid):
    k = "v:" + str(eid)
    v = _VEC_CACHE.get(k)
    if v is not None:
        return v
    try:
        row = c.execute("SELECT data FROM vec WHERE id=?", (eid,)).fetchone()
        if row and row["data"]:
            arr = struct.unpack("<%df" % (len(row["data"]) // 4), row["data"])
            v = list(arr)
            _VEC_CACHE[k] = v
            return v
    except Exception:
        pass
    return None


def _db_vec_put(c, eid, vec):
    try:
        if vec:
            _VEC_CACHE["v:" + str(eid)] = vec
            c.execute("INSERT OR REPLACE INTO vec(id,data) VALUES(?,?)",
                      (eid, struct.pack("<%df" % len(vec), *vec)))
    except Exception:
        pass


def db_recall(req):
    """数据库版召回：SQL 预筛（时间/标签/关键词）→ 词频打分 → 语义 RRF → 强化衰减。"""
    outpath = req.get("index")
    c = _db_ready(req)
    if c is None:
        return None
    if req.get("half_life"):
        try:
            _CFG["half_life"] = float(req.get("half_life"))
        except Exception:
            pass
    q = (req.get("query") or "").strip()
    k = max(1, int(req.get("top_k") or 5))
    terms = tokenize(q)
    if not terms:
        return {"bookmarks": "", "count": 0}
    df, dt, pd = req.get("date_from") or None, req.get("date_to") or None, req.get("period") or None
    if not df and not dt and not pd:
        df, dt, pd = parse_time_hint(q)
    r2 = dict(req)
    r2.update({"date_from": df, "date_to": dt, "period": pd})
    where, args = _db_where_time(r2)
    # ★只给"自动书签"用：按条目日期跳过最近 N 天（不是文件 mtime）；手动工具传 0 不跳过
    try:
        _skip = int(req.get("skip_recent_days") or 0)
    except Exception:
        _skip = 0
    if _skip > 0:
        _cut = (datetime.date.today() - datetime.timedelta(days=_skip)).isoformat()
        where.append("(date IS NULL OR date = '' OR date < ?)")
        args.append(_cut)
    keys = sorted({t for t in terms if len(t) >= 2}, key=len, reverse=True)[:3] or list(terms)[:3]
    like = " OR ".join(["hay LIKE ?"] * len(keys))
    sql = "SELECT * FROM entries WHERE " + " AND ".join(where) + " AND (" + like + ") ORDER BY time DESC LIMIT 400"
    rows = [_row_entry(c, r) for r in c.execute(sql, args + ["%" + t.lower() + "%" for t in keys])]
    if not rows:
        return {"bookmarks": "", "count": 0}
    scored = []
    for e in rows:
        w, hit = _db_score(e, terms)
        if w > 0:
            scored.append((w, hit, e.get("time") or "", e))
    if not scored:
        return {"bookmarks": "", "count": 0}
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    kw_rank = {id(x[3]): i + 1 for i, x in enumerate(scored)}

    fused = None
    qv = _embed([q]) if _MODEL["ready"] else None
    if qv:
        cand = scored[:80]
        sims = []
        for _, _, _, e in cand:
            v = _db_vec_get(c, e.get("id"))
            if v is None:
                vs = _embed([(e.get("summary") or "") + " " + (e.get("text") or "")[:300]]) or []
                v = vs[0] if vs else None
                _db_vec_put(c, e.get("id"), v)
            if v:
                sims.append((_cos(qv[0], v), e))
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
                dual = (id(e) in kw_rank) and (id(e) in sem_rank)
                fuse.append((rrf * _db_boost(e), 1 if dual else 0, e.get("time") or "", e))
            fuse.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            fused = fuse
    final = fused if fused else [(w / 1000.0 * _db_boost(e), hit, t, e) for w, hit, t, e in scored]
    top = final[0][0] if final else 0
    cutoff = top * 0.45
    out = []
    today = datetime.date.today().isoformat()
    for score, _, _, e in final:
        if score < cutoff:
            break
        try:
            c.execute("UPDATE entries SET access_count=access_count+1, last_access=? WHERE id=?", (today, e.get("id")))
        except Exception:
            pass
        out.append(e)
        if len(out) >= k:
            break
    try:
        c.commit()
    except Exception:
        pass
    return {"bookmarks": to_bookmarks(out), "count": len(out)}


def db_browse(req):
    c = _db_ready(req)
    if c is None:
        return None
    where, args = _db_where_time(req)
    q = (req.get("query") or "").strip().lower()
    if q:
        where.append("hay LIKE ?"); args.append("%" + q + "%")
    for key, col in (("who", "people"), ("topic", "topics"), ("thing", "things")):
        v = (req.get(key) or "").strip()
        if v:
            where.append(col + " LIKE ?"); args.append("%" + v + "%")
    wsql = " AND ".join(where)
    total = c.execute("SELECT COUNT(*) FROM entries WHERE " + wsql, args).fetchone()[0]
    off = max(0, int(req.get("offset") or 0))
    lim = max(1, min(200, int(req.get("limit") or 50)))
    rows = c.execute("SELECT * FROM entries WHERE " + wsql + " ORDER BY time DESC LIMIT ? OFFSET ?",
                     args + [lim, off]).fetchall()
    items, lines = [], []
    for r in rows:
        d = _row_entry(c, r)
        d["rel_time"] = rel_time(d.get("date", ""))
        items.append({k2: d.get(k2) for k2 in ("id", "date", "period", "head", "summary", "source", "people", "topics", "things", "rel_time")})
        label = "、".join((d.get("topics") or []) + (d.get("things") or [])) or (d.get("head") or "记忆")
        lines.append("- [%s] %s：%s（出处 %s）" % (d.get("date"), label, d.get("summary"), d.get("source")))
    return {"total": total, "offset": off, "items": items, "text": "\n".join(lines)}


def db_stats(req):
    c = _db_ready(req)
    if c is None:
        return None
    n = c.execute("SELECT COUNT(*) FROM entries WHERE retired=0").fetchone()[0]
    row = c.execute("SELECT MIN(date), MAX(date) FROM entries WHERE retired=0").fetchone()
    first, last = (row[0] or ""), (row[1] or "")
    return {"count": n, "first": first, "last": last,
            "text": "索引共 %d 条，覆盖 %s ~ %s" % (n, first or "?", last or "?")}


def db_links(req):
    c = _db_ready(req)
    if c is None:
        return None
    q = (req.get("query") or "").strip().lower()
    top = max(1, int(req.get("top") or 6))
    want = (req.get("type") or "").strip()
    seeds = []
    if q:
        for r in c.execute("SELECT name,n FROM nodes ORDER BY n DESC LIMIT 500"):
            nl = (r["name"] or "").lower()
            if nl and (nl in q or (len(nl) >= 2 and q in nl)):
                seeds.append((r["name"], r["n"]))
    seeds = seeds[:3]
    items, lines = [], []
    for name, cnt in seeds:
        nb = []
        # ★过滤必须同时作用在 UNION 两段上（否则前半段会把所有边都带出来）
        cond = " AND type LIKE ?" if want else ""
        sql = ("SELECT b AS x,w,type FROM edges WHERE a=?" + cond + " UNION "
               "SELECT a AS x,w,type FROM edges WHERE b=?" + cond + " ORDER BY w DESC LIMIT ?")
        args = [name]
        if want:
            args.append("%" + want + "%")
        args.append(name)
        if want:
            args.append("%" + want + "%")
        args.append(top)
        for r in c.execute(sql, args):
            nb.append({"name": r["x"], "w": r["w"], "type": (r["type"] or "共现").split("|")[0]})
        items.append({"node": name, "count": cnt, "neighbors": nb})
        lines.append("- %s：出现 %d 次；常一起出现 → %s" % (
            name, cnt, "、".join("%s(%d·%s)" % (x["name"], x["w"], x.get("type") or "共现") for x in nb) or "（没有邻居）"))
    return {"seeds": [n for n, _ in seeds], "items": items, "text": "\n".join(lines)}


def db_load_all(outpath):
    """把库内容整份取出（兼容仍按 _load_index() 工作的旧算法，如 compactscan/audit）。"""
    c = _db_for(outpath)
    if c is None:
        return None
    try:
        return [_row_entry(c, r) for r in c.execute("SELECT * FROM entries")]
    except Exception:
        return None


def db_reset_op(req):
    """清空索引库：关连接 → 删 memory.db（官方文件与事实卡都不动）。"""
    idx = req.get("index") or ""
    dbp = os.path.join(os.path.dirname(idx), "memory.db") if idx else ""
    try:
        cn = _DB.get("conn")
        if cn is not None:
            try:
                cn.close()
            except Exception:
                pass
        _DB.update({"ok": False, "path": "", "conn": None})
        if dbp and os.path.isfile(dbp):
            os.remove(dbp)
        return {"count": 0, "text": "索引库已清空（下次 build 会重建）"}
    except Exception as e:
        return {"count": 0, "text": "清空失败：%s" % e}


def db_build_op(req):
    """build 的数据库入口：成功后清掉内存缓存，让后续读取走库。"""
    out = req.get("out") or ""
    n = db_build(req.get("memdir") or "", out, req.get("extra") or [], req.get("exclude") or [])
    if n is None:
        return None
    _set_index(out)
    _IDX["data"] = None
    return {"count": n, "out": out, "text": "建库完成：%d 条" % n}


_DB_OPS = {
    "build": db_build_op,
    "recall": db_recall,
    "browse": db_browse,
    "stats": db_stats,
    "links": db_links,
    "reset": db_reset_op,
}

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
            # ★数据库优先：build/recall/browse/stats/links 直接走 SQLite
            _set_index(req.get("index"))
            _dbop = _DB_OPS.get(fn)
            if _dbop is not None:
                payload = _dbop(req)
                if payload is not None:
                    reply({"id": rid, "ok": True, **payload})
                    continue
            if fn == "ping":
                reply({"id": rid, "ok": True, "device": _MODEL["device"], "dim": 512})
            elif fn == "warm":
                # 预热：后台把索引读进来并建好词表/DF，真查询时就是热的（大库首次冷读会超 200ms 窗口）
                _set_index(req.get("index"))
                _load_index()
                _vocab()
                _df()
                # 再吃掉模型/CUDA 的"首次编码"开销：空转一次 embed，真查询就不带这笔了
                if _MODEL.get("ready"):
                    try:
                        _embed(["预热"])
                    except Exception:
                        pass
                reply({"id": rid, "ok": True, "count": len(_IDX["data"] or [])})
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
                              use_graph=bool(req.get("use_graph", True)),
                              skip_recent_days=int(req.get("skip_recent_days") or 0))
                reply({"id": rid, "ok": True, "bookmarks": to_bookmarks(hits), "count": len(hits)})
            elif fn == "build":
                if req.get("block_max"):
                    _CFG["block_max"] = int(req.get("block_max"))
                memdir = req.get("memdir") or ""
                outpath = req.get("out") or ""
                n, outp = build_index(memdir, outpath, req.get("extra") or [], req.get("exclude") or [])
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
            elif fn == "bytime":
                _set_index(req.get("index"))
                hits = bytime(date_from=req.get("date_from") or None,
                              date_to=req.get("date_to") or None,
                              period=req.get("period") or None,
                              hour_from=req.get("hour_from"),
                              hour_to=req.get("hour_to"),
                              top_k=int(req.get("top_k") or 10))
                reply({"id": rid, "ok": True, "bookmarks": to_bookmarks(hits), "count": len(hits)})
            elif fn == "stats":
                _set_index(req.get("index"))
                st = index_stats()
                txt = "索引共 %d 条，覆盖 %s ~ %s" % (st["count"], st["first"] or "?", st["last"] or "?")
                reply({"id": rid, "ok": True, **st, "text": txt})
            elif fn == "links":
                _set_index(req.get("index"))
                r = links(req.get("query") or "", int(req.get("top") or 6))
                reply({"id": rid, "ok": True, **r})
            elif fn == "bywhen":
                _set_index(req.get("index"))
                r = bywhen(req.get("text") or "", req.get("keyword") or None,
                           int(req.get("top_k") or 8), bool(req.get("use_graph", True)))
                reply({"id": rid, "ok": True, **r})
            elif fn == "compactscan":
                _set_index(req.get("index"))
                cands = compact_scan(req.get("memdir") or "",
                                     int(req.get("min_entries") or 30),
                                     int(req.get("min_age_days") or 7),
                                     int(req.get("max_chars") or 20000),
                                     int(req.get("top_k") or 8),
                                     req.get("exclude") or [])
                reply({"id": rid, "ok": True, "candidates": cands, "count": len(cands)})
            elif fn == "auditscan":
                _set_index(req.get("index"))
                rows = audit_scan(req.get("factsdir") or req.get("memdir") or "", int(req.get("limit") or 30),
                                  float(req.get("dup") or 0.85), float(req.get("near") or 0.5),
                                  int(req.get("max_facts") or 400))
                reply({"id": rid, "ok": True, "issues": rows, "count": len(rows)})
            elif fn == "factmerge":
                r = fact_merge(req.get("lines") or [], req.get("existing") or "", float(req.get("threshold") or 0.5))
                reply({"id": rid, "ok": True, **r})
            elif fn == "browse":
                _set_index(req.get("index"))
                r = browse(req.get("query"), req.get("who"), req.get("topic"), req.get("thing"),
                           req.get("date_from") or None, req.get("date_to") or None,
                           int(req.get("offset") or 0), int(req.get("limit") or 50))
                reply({"id": rid, "ok": True, **r})
            else:
                reply({"id": rid, "ok": False, "error": "unknown fn: %s" % fn})
        except Exception as e:
            reply({"id": rid, "ok": False, "error": "%s: %s" % (type(e).__name__, e)})


if __name__ == "__main__":
    main()
