"""召回、时间查询、画像卡与模型向量支持。"""

import os
import io
import re
import json
import datetime
import hashlib
import sqlite3
import threading
from collections import OrderedDict


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
import profile_view
import semantic_search
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
            v = _vec_cached(None, e)
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
                    _vec_store(None, cand[i][3], vs[j])
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
            # ★必须走持久状态（maint_state.json）+ 重建重放：只 UPDATE 库里的 retired，
            #   下一次建库就会被文件重新投影回来（审查 R1）。这里改成维护模块的统一入口。
            try:
                import maintenance
                factsdir = os.path.dirname(os.path.abspath(_IDX.get("path") or ""))
                if factsdir and os.path.isdir(factsdir):
                    if soft_ids:
                        maintenance.retire_entries(factsdir, _IDX.get("path"), soft_ids,
                                                   reason="daily")
                    if purge_ids:
                        maintenance.purge_entries(factsdir, _IDX.get("path"), purge_ids)
            except Exception as e:
                _warn("persist maintenance decisions", e)
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


def _entry_version(entry):
    """index.json 条目的正文版本，与 dbside 的 text_hash 同一公式（无可核内容返回空）。"""
    body = str(entry.get("text") or entry.get("body") or "")
    if not body:
        return ""
    norm = re.sub(r"\s+", "", body)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def _json_fact_entries(index_path=""):
    """JSON 兜底也要遵守"有效事实"口径：只算 facts/ 来源、未退役的卡。

    数据库不可用时不能把整个 index 直接当画像依据：官方流水、显式 retired 的卡
    都会变成"个人事实"（5.1.1 复核 R09）。维护状态能按 id＋正文版本核对就重放，
    核对不了的降级项必须在画像里说出来，不能假装已恢复。
    """
    rows = [e for e in _index_entries()
            if not e.get("retired") and _is_fact_source(e.get("source"))]
    if not index_path:
        return rows, "降级：未使用插件数据库，画像来自 index.json；维护状态未重放。"
    factsdir = os.path.dirname(os.path.abspath(index_path))
    try:
        import maintenance
        state = maintenance.load_state(factsdir)
    except Exception as exc:
        _warn("read maintenance state for profile fallback", exc)
        return rows, "降级：维护状态不可读，未重放合并／清理决定，结果可能包含已处理条目。"
    # Apply durable manual/audit decisions before considering maintenance tombstones.
    import fact_ops
    import audit_ops
    from textsim import split_tag_list
    try:
        cards, conflicts = fact_ops.projection(index_path)
        audit_state = audit_ops._audit_load_state(index_path)
    except (OSError, RuntimeError, ValueError) as exc:
        _warn("read effective facts for JSON fallback", exc)
        return [], "降级：无法核对人工／审计状态，暂不生成画像；请重建索引。"
    if conflicts:
        return [], "降级：人工操作存在版本冲突，暂不生成画像；请核对并重建索引。"
    effective = []
    for entry in rows:
        entry = dict(entry)
        card = cards.get(str(entry.get("id") or ""))
        if card:
            retired = audit_state.get("retired", {}).get(card["card_id"], {})
            if card["retired"] or (retired and retired.get("version") == card["audit_version"]):
                continue
            if card["edited"]:
                fields = fact_ops._entry_fields(card["body"])
                entry.update(fields)
                for key in ("people", "topics", "things", "shared", "pending"):
                    entry[key] = split_tag_list(fields.get(key))
                entry["text"] = card["body"]
        effective.append(entry)
    rows = effective
    all_rows = [dict(entry, text=cards[str(entry.get("id") or "")]["body"])
                if str(entry.get("id") or "") in cards else entry for entry in _index_entries()]
    by_id = {str(entry.get("id") or ""): entry for entry in all_rows}
    kept, unverifiable, stale_parent = [], 0, 0
    for entry in rows:
        entry_id = str(entry.get("id") or "")
        version = _entry_version(entry)
        purged = (state.get("purged") or {}).get(entry_id)
        if purged is not None:
            want = str((purged or {}).get("version") or "")
            if want and version and want == version:
                continue
            unverifiable += 1
        merged = (state.get("merged") or {}).get(entry_id)
        if merged is not None:
            info = merged or {}
            want = str(info.get("version") or "")
            if not (want and version and want == version):
                unverifiable += 1
            else:
                # 与数据库 apply_state 同口径：父端存在且正文仍是 parent_version，
                # 合并决定才成立；父端缺失/改写时保留子端（5.1.2 复核 S03）。
                parent_id = str(info.get("parent") or "")
                parent_version = str(info.get("parent_version") or version)
                parent = by_id.get(parent_id) if parent_id else None
                if parent is not None and _entry_version(parent) == parent_version:
                    continue
                stale_parent += 1
        retired = (state.get("retired") or {}).get(entry_id)
        if retired is not None:
            want = str((retired or {}).get("version") or "")
            if want and version and want == version:
                continue
            unverifiable += 1
        kept.append(entry)
    notes = []
    if stale_parent:
        notes.append("%d 条合并决定因父端缺失或已改写未执行（子端保留）" % stale_parent)
    if unverifiable:
        notes.append("%d 条决定因缺少可核版本未执行" % unverifiable)
    detail = "；".join(notes) if notes else "维护状态已按 id＋正文版本重放"
    return kept, "降级：未使用插件数据库，画像来自 index.json；" + detail + "。"


def _profile_source(index_path=""):
    """画像取数：有插件库就用库（事实卡口径与页面一致），否则退回 index.json。"""
    if index_path:
        dbp = os.path.join(os.path.dirname(os.path.abspath(index_path)), "memory.db")
        if os.path.isfile(dbp):
            try:
                uri = "file:" + dbp.replace("\\", "/") + "?mode=ro"
                c = sqlite3.connect(uri, uri=True, timeout=5)
                c.row_factory = sqlite3.Row
                try:
                    rows = c.execute(
                        "SELECT * FROM entries WHERE COALESCE(retired,0)=0 "
                        "AND lower(replace(source,char(92),'/')) LIKE 'facts/%'").fetchall()
                    return [_row_entry(c, row) for row in rows], ""
                finally:
                    c.close()
            except sqlite3.Error as exc:
                _warn("read profile facts from database: %s" % dbp, exc)
    return _json_fact_entries(index_path)


def profile_card(who, limit=40, index_path=""):
    """按人聚画像：主题归纳 + 可追溯摘要 + 新→旧证据（PROFILE-01/02/03）。

    旧实现把 `people` 拼成字符串后做子串命中，共同主体与正文提及都会变成"个人事实"，
    再按 time 升序取尾部，于是「昨天」会排在「2 天前」后面。现在改成：
    精确人物标签（含已确认别名）→ 直接事实／共同事件分开 → 按绝对时间倒序 → 主题归纳。
    """
    if not who:
        return ""
    entries, note = _profile_source(index_path)
    import profile_ops
    settings = profile_ops.load(index_path) if index_path else {'aliases': {}, 'hidden': []}
    who = profile_ops.canonical(settings, who)
    if profile_ops.hidden(settings, 'person', who):
        return '该条目已从人物画像列表移除，可在画像管理中恢复；事实卡保留。'
    return profile_view.render(who, entries, limit, note=note, aliases=profile_ops.family(settings, who))


def open_loops(keyword=None, limit=20):
    """钓牵挂与欠账：扫索引里的待办/牵挂词。"""
    kws = [keyword] if keyword else ["牵挂", "欠账", "待办", "还欠", "没动笔", "未解决", "待续", "挂着", "等"]
    rows = []
    for e in _profile_source(_IDX.get("path") or "")[0]:
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


# ---------- 语义缓存预热（后台分批，不阻塞召回） ----------

_WARM = {"running": False, "cancel": False, "total": 0, "done": 0, "cached": 0,
         "started": "", "finished": "", "error": "", "batch": 64, "index": ""}
_WARM_THREAD = {"t": None}


def vec_stats():
    """语义状态与缓存覆盖，供 memstatus / 页面查看。"""
    return {
        "model": _model_id(),
        "ready": bool(_MODEL.get("ready")),
        "warmed": bool(_MODEL.get("warmed")),
        "device": str(_MODEL.get("device") or ""),
        "error": str(_MODEL.get("err") or _WARM["error"] or ""),
        "running": bool(_WARM["running"]),
        "total": int(_WARM["total"]),
        "done": int(_WARM["done"]),
        "cached": int(_WARM["cached"]),
        "started": _WARM["started"],
        "finished": _WARM["finished"],
        "mem_cache": len(_VEC_CACHE),
        "reranker": semantic_search.reranker_status(),
    }


def _vec_warm_loop(index_path, batch, limit):
    conn = None
    try:
        if not index_path:
            raise RuntimeError("index path is required for warmup")
        dbp = os.path.join(os.path.dirname(os.path.abspath(index_path)), "memory.db")
        if not os.path.isfile(dbp):
            raise RuntimeError("memory.db not found")
        conn = dbside._db_open_aux(dbp)
        model = _model_id()
        have = {str(r["id"]): str(r["version"] or "")
                for r in conn.execute("SELECT id,version FROM vec WHERE model=?", (model,))}
        rows = conn.execute(
            "SELECT id,text_hash,summary,body FROM entries WHERE COALESCE(retired,0)=0 "
            "AND lower(replace(source,char(92),'/')) LIKE 'facts/%' "
            "ORDER BY date DESC,time DESC,id").fetchall()
        todo = []
        for row in rows:
            eid = str(row["id"])
            version = str(row["text_hash"] or "")
            if have.get(eid) != version:      # 正文改过 / 没算过 → 要重算
                todo.append((eid, version, str(row["summary"] or ""), str(row["body"] or "")[:300]))
        _WARM["cached"] = max(0, len(rows) - len(todo))
        if limit > 0:
            todo = todo[:limit]
        _WARM["total"] = len(todo)
        step = max(1, int(batch))
        for start in range(0, len(todo), step):
            if _WARM["cancel"]:
                break
            chunk = todo[start:start + step]
            vecs = _embed([(item[2] + " " + item[3]).strip() for item in chunk])
            if not vecs or len(vecs) != len(chunk) or any(not vec for vec in vecs):
                raise RuntimeError(str(_MODEL.get("err") or "embedding batch incomplete"))
            for (eid, version, _s, _b), vec in zip(chunk, vecs):
                if vec:
                    _db_vec_put(conn, eid, vec, version, model, commit=False)
            conn.commit()
            _WARM["done"] = min(len(todo), start + len(chunk))
    except Exception as e:
        _WARM["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        _WARM["running"] = False
        _WARM["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        _WARM_THREAD["t"] = None


def vec_warm_start(index_path="", batch=64, limit=0):
    """后台分批预热：重复调用幂等，正在跑时只回状态。"""
    if _WARM["running"]:
        return vec_stats()
    _WARM.update({"running": True, "cancel": False, "total": 0, "done": 0, "cached": 0,
                  "started": datetime.datetime.now().isoformat(timespec="seconds"),
                  "finished": "", "error": "", "batch": max(1, int(batch or 64)),
                  "index": str(index_path or "")})
    t = threading.Thread(target=_vec_warm_loop,
                         args=(index_path, int(batch or 64), int(limit or 0)),
                         name="memoryinject-vec-warm", daemon=True)
    _WARM_THREAD["t"] = t
    t.start()
    return vec_stats()


def vec_warm_stop():
    """请求停止；当前批次跑完就退出（不打断一次编码）。"""
    _WARM["cancel"] = True
    return vec_stats()


def index_stats():
    """索引体量与时间跨度 + 维护记录 + 语义缓存，供 memstatus 用。"""
    data = [e for e in indexstore._load_index() if not e.get("retired")]
    dates = sorted({str(e.get("date") or "") for e in data if e.get("date")})
    return {"count": len(data), "first": dates[0] if dates else "", "last": dates[-1] if dates else "",
            "maintenance": dbside.maintenance_status(5), "semantic": vec_stats()}


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
    """向量化；失败返回 None（调用方一律退回纯词频）。

    带锁：后台预热线程与召回查询会同时用到同一个模型，torch 模型不是线程安全的。
    """
    try:
        with _MODEL_LOCK:
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


# ---------- 语义缓存：键 = 条目 + 正文版本 + 模型标识 ----------

_MODEL_LOCK = threading.Lock()
_MODEL_ID = {"value": ""}


def _model_id():
    """模型标识：完整路径 + 配置文件的修订（mtime/大小），不是只取末级目录名。

    只取 basename 时，`C:/models/A/checkpoint` 与 `C:/models/B/checkpoint` 会撞在一起，
    换模型后读到旧向量、不同向量空间混用（审查 R6）。
    """
    if _MODEL_ID["value"]:
        return _MODEL_ID["value"]
    path = ""
    try:
        path = find_model_dir() or ""
    except Exception as e:
        _warn("identify embedding model", e)
    if not path:
        _MODEL_ID["value"] = "unknown"
        return _MODEL_ID["value"]
    full = os.path.normcase(os.path.abspath(path))
    revision = ""
    try:
        stat = os.stat(os.path.join(path, "config.json"))
        revision = "|%d:%d" % (int(stat.st_mtime), int(stat.st_size))
    except OSError:
        pass
    digest = hashlib.sha1((full + revision).encode("utf-8")).hexdigest()[:12]
    _MODEL_ID["value"] = "%s@%s" % (os.path.basename(os.path.normpath(path)), digest)
    return _MODEL_ID["value"]


# Process-local query vectors are reusable across characters for the same model.
_QUERY_VECTORS = OrderedDict()
_QUERY_VECTOR_MAX = 64


def _query_vector(query):
    key = (_model_id(), id(_MODEL.get("mdl")), query)
    if key in _QUERY_VECTORS:
        _QUERY_VECTORS.move_to_end(key)
        return _QUERY_VECTORS[key]
    text = "为这个句子生成表示以用于检索相关文章：" + query if "bge-small-zh" in key[0] else query
    vectors = _embed([text])
    if not vectors or not vectors[0]:
        return None
    result = vectors[0]
    _QUERY_VECTORS[key] = result
    while len(_QUERY_VECTORS) > _QUERY_VECTOR_MAX:
        _QUERY_VECTORS.popitem(last=False)
    return result


def _vec_meta(entry):
    return (str(entry.get("text_hash") or entry.get("version") or ""), _model_id())


def _vec_cached(c, entry):
    version, model = _vec_meta(entry)
    return _db_vec_get(c, str(entry.get("id") or ""), version, model)


def _vec_store(c, entry, vec):
    version, model = _vec_meta(entry)
    _db_vec_put(c, str(entry.get("id") or ""), vec, version, model)


def _entry_vector(entry):
    """单条记忆的向量（进程内 + 库缓存）。"""
    v = _vec_cached(None, entry)
    if v is not None:
        return v
    vs = _embed([_field_text(entry, "summary") + " " + _field_text(entry, "text")[:300]])
    if not vs:
        return None
    _vec_store(None, entry, vs[0])
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


def conservative_semantic_order(scored, semantic):
    ordered = [(w / 1000.0 * _db_boost(e), hit, tm, e) for w, hit, tm, e in scored]
    ordered.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    for i in range(1, len(ordered)):
        j = i
        while j > 0:
            current, previous = ordered[j], ordered[j - 1]
            # No boost, no crossing clear keyword differences, no noisy semantic tie-break.
            if current[0] < previous[0] * .95:
                break
            cs, ps = semantic.get(id(current[3])), semantic.get(id(previous[3]))
            if cs is None or ps is None or cs - ps < .05:
                break
            ordered[j - 1], ordered[j] = current, previous
            j -= 1
    return ordered


def evidence_constraints(query, entries):
    from textsim import classified_people, split_tag_list
    import profile_view
    q = query.casefold()
    people, topics = set(), set()
    for entry in entries:
        if entry.get("retired") or not _is_fact_source(entry.get("source")):
            continue
        people.update(name for name in classified_people(entry) if len(name) >= 2 and name.casefold() in q)
        topics.update(name for name in split_tag_list(entry.get("topics")) if len(name) >= 2 and name.casefold() in q)
    # Longest matching names take precedence over an embedded short alias.
    people = {name for name in people if not any(name != other and name in other for other in people)}
    wanted = {alias.casefold() for name in people for alias in profile_view.aliases_of(name)}
    def accepts(entry):
        if wanted and not wanted.intersection(name.casefold() for name in classified_people(entry)):
            return False
        if topics:
            row_topics = {name.casefold() for name in split_tag_list(entry.get("topics"))}
            if not topics.intersection(row_topics):
                # ★标签不是全覆盖：同一件事可能只挂了"健康"而没挂"血压"。
                #   查询词本身就是主题词时，正文里出现过也算命中，
                #   否则「我前天血压多少」会因为候选卡只标健康而全军覆没。
                body = " ".join(str(entry.get(key) or "") for key in ("head", "summary", "text")).casefold()
                if not any(topic in body for topic in topics):
                    return False
        return True
    return accepts


def db_recall(req):
    """数据库版召回：关键词与有效事实向量候选 → 可选本地精排 → 去重与回执。

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
    # ★预筛词按 tokenize 的输出顺序取前几个，不能对同长度集合做 sorted：
    #   全 2 字词时集合顺序随进程哈希变化，可能只取到"线是/是怎"这类碎片，
    #   把真正含"东侧航线"的条目全部漏掉（表现为同一句话召回时有时无）。
    # ★长句整串（>6 字）不能当预筛词：tokenize 会先把 5–8 字的整句当一个词，
    #   "我前天血压多少"的前三个词就是「整句/我前/前天」，真正的「血压」被挤掉，
    #   结果候选为 0。整串只在短实体名（≤6 字）时才保留。
    keys = []
    for term in terms:
        if len(term) >= 2 and len(term) <= 6 and term not in keys:
            keys.append(term)
        if len(keys) >= 5:
            break
    if not keys:
        keys = [t for t in terms if len(t) >= 2][:3] or list(terms)[:3]
    # DB-01：hay 只存标题+摘要+标签，正文只留 body 一份；预筛同时看两列，
    # 正文仍可命中（LIKE 对 ASCII 依旧大小写不敏感），检索契约不变。
    like = " OR ".join(["hay LIKE ?", "body LIKE ?"] * len(keys))
    like_args = ["%" + t.lower() + "%" for t in keys for _ in (0, 1)]
    sql = "SELECT * FROM entries WHERE " + " AND ".join(where) + " AND (" + like + ") ORDER BY time DESC LIMIT 400"
    rows = [_row_entry(c, r) for r in c.execute(sql, args + like_args)]
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
            like2 = " OR ".join(["hay LIKE ?", "body LIKE ?"] * len(keys2))
            like2_args = ["%" + t.lower() + "%" for t in keys2 for _ in (0, 1)]
            sql2 = "SELECT * FROM entries WHERE " + " AND ".join(where) + " AND (" + like2 + ") ORDER BY time DESC LIMIT 200"
            rows = [_row_entry(c, r) for r in c.execute(sql2, args + like2_args)]
            score_terms = list(dict.fromkeys(list(terms) + [t.lower() for t in keys2]))
    accepts = evidence_constraints(q, indexstore._load_index())
    # Expanded graph terms cannot replace evidence present in the actual question.
    rows = [e for e in rows if accepts(e) and _db_score(e, terms)[0] > 0]
    scored = []
    for e in rows:
        w, hit = _db_score(e, score_terms)
        if w > 0:
            scored.append((w, hit, e.get("time") or "", e))
    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    kw_rank = {id(x[3]): i + 1 for i, x in enumerate(scored)}

    fused = None
    use_semantic = bool(req.get("use_semantic", False))
    try:
        semantic_min_score = float(req.get("semantic_min_score", 0.58))
    except (ValueError, TypeError) as e:
        semantic_min_score = 0.58
    qv = _query_vector(q) if use_semantic and (_MODEL["ready"] and _MODEL.get("warmed")) else None
    semantic_candidates, vector_coverage = [], {"cached": 0, "total": 0, "complete": False}
    if qv:
        try:
            vector_minimum = max(0.0, min(1.0, float(req.get('vector_min_score', .4))))
        except (TypeError, ValueError):
            vector_minimum = .4
        try:
            semantic_candidates, vector_coverage = semantic_search.vector_candidates(
                c, _model_id(), qv, where, args, accepts, minimum=vector_minimum)
        except Exception as exc:
            _warn("search fact vectors", exc)
    if qv:
        cand = scored[:80]
        # ★批量编码：逐条 _embed 在 80 个候选时要 15 秒+（首次召回必超时），
        #   一次 batch 提交只要零点几秒
        vmap = {}
        need = []
        for _, _, _, e in cand:
            v = _vec_cached(c, e)
            if v is not None:
                vmap[e.get("id")] = v
            else:
                need.append(e)
        if need:
            texts = [((x.get("summary") or "") + " " + (x.get("text") or "")[:300]) for x in need]
            batch = _embed(texts) or []
            wrote = 0
            for x, v in zip(need, batch):
                if v:
                    vmap[x.get("id")] = v
                    # 批量写：一批只提交一次，避免逐条 commit
                    _db_vec_put(c, str(x.get("id") or ""), v, *_vec_meta(x), commit=False)
                    wrote += 1
            # ★前台召回自己收事务：只 INSERT 不提交会一直占着写锁，
            #   别的连接读不到、后台预热也写不进去（审查 R2）。
            if wrote:
                try:
                    c.commit()
                except sqlite3.Error as e:
                    _warn("commit vector cache", e)
                    try:
                        c.rollback()
                    except sqlite3.Error as rollback_error:
                        _warn("rollback vector cache", rollback_error)
        sims = []
        for _, _, _, e in cand:
            v = vmap.get(e.get("id"))
            if v:
                sim = _cos(qv, v)
                if sim >= semantic_min_score:
                    sims.append((sim, e))
        if sims:
            sims.sort(key=lambda x: x[0], reverse=True)
            sem_score = {id(e): s for s, e in sims}
            # Semantic can only swap adjacent, lexically near-tied candidates.
            fused = conservative_semantic_order(scored, sem_score)
    final = fused if fused else [(w / 1000.0 * _db_boost(e), hit, t, e) for w, hit, t, e in scored]
    # Balance the bounded pool: lexical candidates cannot crowd out semantic-only hits.
    pool, seen_ids = [], set()
    keyword_entries = [item[3] for item in final]
    vector_entries = [entry for _, entry in semantic_candidates]
    for entry in keyword_entries[:12] + vector_entries[:12] + keyword_entries + vector_entries:
        if entry['id'] not in seen_ids:
            pool.append(entry)
            seen_ids.add(entry['id'])
        if len(pool) >= 24:
            break
    rerank_scores = None
    if req.get('use_reranker', False) and pool:
        rerank_scores = semantic_search.rerank(q, pool, str(req.get('rerank_model_path') or ''))
    if rerank_scores is not None:
        try:
            rerank_minimum = max(0.0, min(1.0, float(req.get('rerank_min_score', .1))))
        except (TypeError, ValueError):
            rerank_minimum = .1
        final = [(rerank_scores[entry['id']], 0, entry.get('time') or '', entry) for entry in pool
                 if rerank_scores.get(entry['id'], -1) >= rerank_minimum]
        final.sort(key=lambda row: (row[0], row[2], str(row[3]['id'])), reverse=True)
    # If the reranker is unavailable/disabled, semantic-only hits are not silently injected.
    top = max((row[0] for row in final), default=0)
    cutoff = top * 0.45
    out = []
    seen_sources = {}   # ★必须是 dict：下面用 .get(source, 0) 计数（曾误写成 set()，导致召回必抛 AttributeError）
    seen_dates = {}
    seen_text = set()   # 同一条内容被复制到多天时，只出一条
    for score, _, _, e in final:
        if score < cutoff:
            continue
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
        "vector_candidates": len(semantic_candidates), "vector_coverage": vector_coverage,
        "reranked": rerank_scores is not None, "reranker": semantic_search.reranker_status(),
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
    coverage = d.get("vector_coverage") or {}
    text += "；全库向量覆盖 %s/%s；精排 %s" % (coverage.get("cached", 0), coverage.get("total", 0),
                                                  "已执行" if d.get("reranked") else "未执行/降级")
    return {"text": text, "debug": d, "bookmarks": result.get("bookmarks", "")}
