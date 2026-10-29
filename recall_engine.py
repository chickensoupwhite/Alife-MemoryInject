"""召回、时间查询、画像卡与模型向量支持。"""

import os
import io
import re
import json
import datetime
import hashlib


# >>> WP1 导入（自动生成，勿手改）
import dbside
import recall_policy
from dbside import (
    _db_boost,
    _db_score,
    _db_vec_get,
    _db_vec_put,
    _db_where_time,
    _fact_references,
    _is_excluded,
    _is_fact_source,
    _row_entry,
)
from graph_indexer import graph_expand
import indexstore
from indexstore import _idf
from state import (
    HARD_DELETE_DAYS,
    SOFT_DELETE_DAYS,
    W_HEAD,
    W_SUMMARY,
    W_TAG,
    W_TEXT,
    _CFG,
    _DB,
    _HERE,
    _IDX,
    _LIFE,
    _MODEL,
    _VEC_CACHE,
    _find_index,
    _warn,
)
from textsim import _entry_weight, _field_text, _looks_important, parse_time_hint, rel_time, score_entry
# <<< WP1 导入

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


def recall(query, top_k=5, min_ratio=0.45, date_from=None, date_to=None, period=None,
           hour_from=None, hour_to=None, use_graph=True, skip_recent_days=0, use_semantic=False):
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
    terms = indexstore.tokenize(query)
    if not terms:
        return []
    data = [e for e in indexstore._load_index() if not e.get("retired") and _is_fact_source(e.get("source"))]
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
    # ★未预热时首次 _embed 会吃 15s CUDA 预热，别让它卡在用户查询里
    # ★语义开关默认关：JSON 兜底路径也必须遵守，否则 memwhen+keyword 会偷偷跑向量
    qv = _embed([query]) if (use_semantic and _MODEL["ready"] and _MODEL.get("warmed")) else None
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


def _touch(entry):
    """Recall selection is not confirmed delivery; never mutate call/lifecycle stats here."""
    return None


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
    except Exception as e:
        _warn('flush life state to index', e)
        pass


def maintain_index(memdir=None, now=None, allow_soft=False):
    """每日维护：软删(标记 retired) → 物理清。带"错过天数补偿"——久没跑一次跑到位。
    ★软删默认关闭（allow_soft=False）：只有显式开启，才把"从未访问 + 老过60天 +
      不像牵挂/约定"的条目标记 retired；物理清只针对已 retired 且够久的。
    返回 {scanned, soft_deleted, purged}。不改文件，只改 index.json。"""
    data = indexstore._load_index()
    today = now or datetime.date.today()
    soft = purged = 0
    keep = []
    soft_ids, purge_ids = [], []   # ★DB 模式要按 id 落库，不能只改内存
    for e in data:
        # 已经软删的：看多久了，够久就物理清
        if e.get("retired"):
            try:
                rd = datetime.date.fromisoformat(str(e.get("retired_at"))[:10])
            except ValueError as e:
                rd = today
            if (today - rd).days >= HARD_DELETE_DAYS:
                purged += 1
                purge_ids.append(e.get("id"))
                continue
            keep.append(e)
            continue
        # 判定软删：从没被访问过、且老过 SOFT_DELETE_DAYS、且衰减很低
        last = e.get("last_access") or ""
        ac = int(e.get("access_count") or 0)
        try:
            d = datetime.date.fromisoformat(str(e.get("date") or "")[:10])
        except ValueError as e:
            d = today
        age = (today - d).days
        if allow_soft and ac == 0 and age >= SOFT_DELETE_DAYS and not last and not _looks_important(e):
            e["retired"] = True
            e["retired_at"] = today.isoformat()
            soft += 1
            soft_ids.append(e.get("id"))
        keep.append(e)
    if soft or purged:
        c = dbside._db_for(_IDX.get("path"))
        if c is not None:
            # ★数据库模式：直接 UPDATE/DELETE，写 JSON 没用（下次读库就丢了）
            try:
                if soft_ids:
                    c.executemany("UPDATE entries SET retired=1, retired_at=? WHERE id=?",
                                  [(today.isoformat(), i) for i in soft_ids if i])
                if purge_ids:
                    c.executemany("DELETE FROM entries WHERE id=?", [(i,) for i in purge_ids if i])
                c.commit()
            except Exception as e:
                _warn('rewrite index file', e)
                pass
            _IDX["data"] = None
        else:
            p2 = _IDX.get("path") or _find_index()
            if p2:
                io.open(p2, "w", encoding="utf-8").write(json.dumps(keep, ensure_ascii=False, indent=1))
                _IDX["data"] = None
    return {"scanned": len(data), "soft_deleted": soft, "purged": purged}


def render_view(outpath=None):
    """生成索引浏览页：每条 相对时间/人事物/摘要/出处，按时间倒序。"""
    import html
    data = indexstore._load_index()
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


def _index_entries():
    return indexstore._load_index()


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


def bytime(date_from=None, date_to=None, period=None, hour_from=None, hour_to=None, top_k=10):
    """按时间范围列记忆（不要求关键词）：日/时段/小时都能下钻，按时间倒序返回。"""
    data = [e for e in indexstore._load_index() if not e.get("retired") and _is_fact_source(e.get("source"))]

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


def bywhen(text, keyword=None, top_k=8, use_graph=True, use_semantic=False):
    """一句中文时间词 -> 那段时间的记忆。「上周三」「三天前」「昨晚」都行；给关键词就按相关度排。"""
    df, dt, pd = parse_time_hint(text or "")
    if keyword:
        hits = recall(keyword, top_k=top_k, date_from=df, date_to=dt, period=pd, use_graph=use_graph,
                      use_semantic=use_semantic)
    else:
        hits = bytime(date_from=df, date_to=dt, period=pd, top_k=top_k)
    return {
        "bookmarks": to_bookmarks(hits),
        "count": len(hits),
        "facts": _fact_references(hits),
        "date_from": df or "",
        "date_to": dt or "",
        "period": pd or "",
        "text": "%s ~ %s%s 共 %d 条" % (df or "?", dt or "?", (" " + pd) if pd else "", len(hits)),
    }


def index_stats():
    """索引体量与时间跨度，供 memstatus 用。"""
    data = [e for e in indexstore._load_index() if not e.get("retired")]
    dates = sorted({str(e.get("date") or "") for e in data if e.get("date")})
    return {"count": len(data), "first": dates[0] if dates else "", "last": dates[-1] if dates else ""}


def compact_scan(memdir, min_entries=30, min_age_days=7, max_chars=20000, top_k=8, exclude=None):
    """找可压缩的流水文件：条目多、够旧。返回候选（含要喂给模型的正文）。
    ★不动原文：这里只给候选，C# 侧提炼、归档、重建索引。"""
    groups = {}
    for e in indexstore._load_index():
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
            except ValueError as e:
                age = 0
        if age < int(min_age_days):
            continue
        items.sort(key=lambda x: str(x.get("time") or ""))
        parts = []
        provenance = []
        for x in items:
            who = "".join(x.get("people") or [])
            topic = "".join(x.get("topics") or [])
            summary = str(x.get("summary") or x.get("head") or "").strip()
            parts.append(" | ".join([str(x.get("date") or ""), who, topic, summary]))
            body = str(x.get("text") or x.get("body") or summary)
            ref = {"scope": "input-range", "source": src,
                   "entry_id": str(x.get("id") or ""),
                   "anchor": str(x.get("anchor") or ""),
                   "content_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                   "excerpt": body[:1600], "excerpt_truncated": len(body) > 1600}
            # Preserve an offset only when the index explicitly provides one;
            # never infer offsets from summaries or concatenated model input.
            if x.get("offset") is not None:
                ref["offset"] = x.get("offset")
            provenance.append(ref)
        size = sum(len(p) + 1 for p in parts)
        if int(max_chars) <= 0:
            # probe 模式：只数候选，不抓正文（预览用）
            cands.append({"source": src, "entries": len(items), "chars": size,
                          "date_from": dates[0] if dates else "", "date_to": dates[-1] if dates else "",
                          "text": "", "provenance": provenance})
            continue
        # ★整份放不进预算就不压：半份压缩后会把来源标记"已压缩"（不再进索引），
        #   尾巴那部分事实既没进事实卡、又从索引消失 = 静默丢数据
        if size > int(max_chars):
            continue
        cands.append({
            "source": src,
            "entries": len(items),
            "chars": size,
            "date_from": dates[0] if dates else "",
            "date_to": dates[-1] if dates else "",
            "text": "\n".join(parts), "provenance": provenance,
        })
    cands.sort(key=lambda c: (-c["entries"], c["date_from"]))
    return cands[:max(1, int(top_k))]


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
    except Exception as e:
        _warn('load embedding model', e)
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
    except (TypeError, ValueError, ZeroDivisionError) as e:
        return 0.0


def db_bywhen(req):
    text = req.get("text") or ""
    keyword = (req.get("keyword") or "").strip()
    try:
        top_k = max(1, min(200, int(req.get("top_k") or 8)))
    except (TypeError, ValueError):
        top_k = 8
    date_from, date_to, period = parse_time_hint(text)
    if keyword:
        result = db_recall(dict(req, query=keyword, top_k=top_k, date_from=date_from,
                                date_to=date_to, period=period, record_calls=False,
                                entry="memwhen"))
        if result is None:
            return None
        bookmarks = result.get("bookmarks", "")
        facts = result.get("facts", [])
        count = result.get("count", 0)
    else:
        c = dbside._db_ready(req)
        if c is None:
            return None
        where, args = _db_where_time({"date_from": date_from, "date_to": date_to, "period": period})
        where.append("lower(replace(source,char(92),'/')) LIKE 'facts/%'")
        rows = [_row_entry(c, row) for row in c.execute(
            "SELECT * FROM entries WHERE " + " AND ".join(where) + " ORDER BY time DESC LIMIT ?",
            args + [top_k]).fetchall()]
        bookmarks = to_bookmarks(rows)
        facts = _fact_references(rows)
        count = len(rows)
    return {
        "bookmarks": bookmarks,
        "count": count,
        "facts": facts,
        "date_from": date_from or "",
        "date_to": date_to or "",
        "period": period or "",
        "text": "%s ~ %s%s 共 %d 条" % (date_from or "?", date_to or "?", (" " + period) if period else "", count),
    }


def db_recall(req):
    """数据库版召回：SQL 预筛（时间/标签/关键词）→ 词频打分 → 语义 RRF → 强化衰减。

    Selection never writes delivery statistics. The caller must explicitly
    acknowledge the returned fact references after crossing its return boundary.
    """
    outpath = req.get("index")
    c = dbside._db_ready(req)
    if c is None:
        return None
    if req.get("half_life"):
        try:
            _CFG["half_life"] = float(req.get("half_life"))
        except (ValueError, TypeError) as e:
            pass
    q = (req.get("query") or "").strip()
    k = max(1, int(req.get("top_k") or 5))
    # D10：门控与压制只在侧车、只作用于自动召回；主动工具永远放行。
    automatic = not recall_policy.is_active_tool(req)
    gate_on = automatic and recall_policy.switch(req, "recall_gate")
    filter_on = automatic and recall_policy.switch(req, "recall_filter")
    # ★话题与窗口必须在门控、空词表等任何提前返回之前推进：
    #   否则"被门控拦掉"和"没有候选"的话题都不会被记录下来，旧窗口会一直沿用。
    terms = indexstore.tokenize(q)
    window = None
    if filter_on:
        window = recall_policy.prepare(recall_policy.session_of(req), terms, True)
    if gate_on:
        allowed, gate_reason = recall_policy.gate_allows(q)
        if not allowed:
            return {"bookmarks": "", "count": 0, "facts": [],
                    "window_gen": window["gen"] if window else 0,
                    "debug": {"query": q, "gate": gate_reason, "skipped": True, "returned": 0}}
    if not terms:
        return {"bookmarks": "", "count": 0, "facts": [],
                "window_gen": window["gen"] if window else 0}
    df, dt, pd = req.get("date_from") or None, req.get("date_to") or None, req.get("period") or None
    if not df and not dt and not pd:
        df, dt, pd = parse_time_hint(q)
    r2 = dict(req)
    r2.update({"date_from": df, "date_to": dt, "period": pd})
    where, args = _db_where_time(r2)
    where.append("lower(replace(source,char(92),'/')) LIKE 'facts/%'")
    # ★只给"自动书签"用：按条目日期跳过最近 N 天（不是文件 mtime）；手动工具传 0 不跳过
    try:
        _skip = int(req.get("skip_recent_days") or 0)
    except (ValueError, TypeError) as e:
        _skip = 0
    if _skip > 0:
        _cut = (datetime.date.today() - datetime.timedelta(days=_skip)).isoformat()
        where.append("(date IS NULL OR date = '' OR date < ?)")
        args.append(_cut)
    keys = sorted({t for t in terms if len(t) >= 2}, key=len, reverse=True)[:3] or list(terms)[:3]
    like = " OR ".join(["hay LIKE ?"] * len(keys))
    sql = "SELECT * FROM entries WHERE " + " AND ".join(where) + " AND (" + like + ") ORDER BY time DESC LIMIT 400"
    rows = [_row_entry(c, r) for r in c.execute(sql, args + ["%" + t.lower() + "%" for t in keys])]
    score_terms = terms
    if not rows and req.get("use_graph", True):
        # ★图谱兜底：关键词一个都没命中时，顺关系链捞（旧 JSON 路径有，DB 版不能丢）
        try:
            gexp = graph_expand(q, terms) or []
        except Exception as e:
            _warn('expand graph keywords', e)
            gexp = []
        keys2 = sorted({str(t) for t in gexp if str(t).strip()}, key=len, reverse=True)[:5]
        if keys2:
            like2 = " OR ".join(["hay LIKE ?"] * len(keys2))
            sql2 = "SELECT * FROM entries WHERE " + " AND ".join(where) + " AND (" + like2 + ") ORDER BY time DESC LIMIT 200"
            rows = [_row_entry(c, r) for r in c.execute(sql2, args + ["%" + t.lower() + "%" for t in keys2])]
            score_terms = list(dict.fromkeys(list(terms) + [t.lower() for t in keys2]))
    if not rows:
        return {"bookmarks": "", "count": 0}
    scored = []
    for e in rows:
        w, hit = _db_score(e, score_terms)
        if w > 0:
            scored.append((w, hit, e.get("time") or "", e))
    if not scored:
        return {"bookmarks": "", "count": 0}
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    kw_rank = {id(x[3]): i + 1 for i, x in enumerate(scored)}

    fused = None
    use_semantic = bool(req.get("use_semantic", False))
    try:
        semantic_min_score = float(req.get("semantic_min_score", 0.58))
    except (ValueError, TypeError) as e:
        semantic_min_score = 0.58
    qv = _embed([q]) if use_semantic and (_MODEL["ready"] and _MODEL.get("warmed")) else None
    if qv:
        cand = scored[:80]
        # ★批量编码：逐条 _embed 在 80 个候选时要 15 秒+（首次召回必超时），
        #   一次 batch 提交只要零点几秒
        vmap = {}
        need = []
        for _, _, _, e in cand:
            v = _db_vec_get(c, e.get("id"))
            if v is not None:
                vmap[e.get("id")] = v
            else:
                need.append(e)
        if need:
            texts = [((x.get("summary") or "") + " " + (x.get("text") or "")[:300]) for x in need]
            batch = _embed(texts) or []
            for x, v in zip(need, batch):
                if v:
                    vmap[x.get("id")] = v
                    _db_vec_put(c, x.get("id"), v)
        sims = []
        for _, _, _, e in cand:
            v = vmap.get(e.get("id"))
            if v:
                sim = _cos(qv[0], v)
                if sim >= semantic_min_score:
                    sims.append((sim, e))
        if sims:
            sims.sort(key=lambda x: x[0], reverse=True)
            sem_score = {id(e): s for s, e in sims}
            # ★语义只能做"最多 +25%"的加权微调：
            #   原来的 0.35 * 1/(60+rank) 是加法项，量级能压过 kw/1000，
            #   实测开语义后"血压"前五会整体换人，等于语义在主导排序。
            #   改成乘性提升，关键词顺序仍是主体，语义只把高相似候选往上抬一点。
            fuse = []
            for kw, hit, tm, e in scored:
                score = (kw / 1000.0) * _db_boost(e)
                sim = sem_score.get(id(e))
                if sim is not None:
                    span = max(1e-6, 1.0 - semantic_min_score)
                    lift = max(0.0, min(1.0, (sim - semantic_min_score) / span))
                    score *= (1.0 + 0.25 * lift)
                fuse.append((score, hit, tm, e))
            fuse.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            fused = fuse
    final = fused if fused else [(w / 1000.0 * _db_boost(e), hit, t, e) for w, hit, t, e in scored]
    top = final[0][0] if final else 0
    cutoff = top * 0.45
    out = []
    seen_sources = {}   # ★必须是 dict：下面用 .get(source, 0) 计数（曾误写成 set()，导致召回必抛 AttributeError）
    seen_dates = {}
    seen_text = set()   # 同一条内容被复制到多天时，只出一条
    for score, _, _, e in final:
        if score < cutoff:
            break
        source = e.get("source") or ""
        date = e.get("date") or ""
        # 多样性：同一出处最多两条、同一天最多三条，避免连续切块淹没真正不同的记忆。
        if seen_sources.get(source, 0) >= 2 or seen_dates.get(date, 0) >= 3:
            continue
        _tkey = re.sub(r"\s+", "", e.get("text") or e.get("summary") or "")
        if _tkey and _tkey in seen_text:
            continue
        if _tkey:
            seen_text.add(_tkey)
        out.append(e)
        seen_sources[source] = seen_sources.get(source, 0) + 1
        seen_dates[date] = seen_dates.get(date, 0) + 1
        if len(out) >= k:
            break
    recall_policy.remember_returned(recall_policy.session_of(req), out)
    suppressed = 0
    if filter_on and window is not None:
        # 断层已在上方完成；这里只用同一窗口的指纹压制，不再推进代次
        out, blocked = recall_policy.suppress(out, window)
        suppressed = len(blocked)
    return {"bookmarks": to_bookmarks(out), "count": len(out), "facts": _fact_references(out), "debug": {
        "query": q, "candidates": len(scored), "semantic": bool(qv), "returned": len(out),
        "cutoff": cutoff, "skipped_recent": int(req.get("skip_recent_days") or 0),
        "gate": "open" if gate_on else ("bypass" if not automatic else "off"),
        "filter": "on" if filter_on else ("bypass" if not automatic else "off"),
        "suppressed": suppressed, "window_gen": window["gen"] if window else 0,
    }, "window_gen": window["gen"] if window else 0}


def db_debug(req):
    # Debugging is candidate inspection, not a delivered recall.  Keep it out
    # of call_events/access_count while sharing the exact ranking path.
    result = db_recall(dict(req, top_k=max(1, min(20, int(req.get("top_k") or 5))), record_calls=False))
    if result is None: return None
    d = result.get("debug") or {}
    text = "查询：%s；候选：%s；模式：%s；最终返回：%s；阈值：%.4f；最近跳过：%s 天" % (
        d.get("query", ""), d.get("candidates", 0), "关键词+语义" if d.get("semantic") else "关键词/图谱",
        d.get("returned", 0), float(d.get("cutoff", 0)), d.get("skipped_recent", 0))
    return {"text": text, "debug": d, "bookmarks": result.get("bookmarks", "")}
