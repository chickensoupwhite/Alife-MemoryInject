"""SQLite 连接、建库、迁移、行映射与关系投影；统计表结构的建立与聚合。"""

import os
import io
import re
import json
import datetime
import hashlib
import sqlite3
import struct


# >>> WP1 导入（自动生成，勿手改）
from state import W_HEAD, W_SUMMARY, W_TAG, W_TEXT, _DB, _DB_SCHEMA, _VEC_CACHE, _set_index, _warn
from textsim import _entry_weight, extract_entities
# <<< WP1 导入
def _is_meta_file(name):
    """是不是插件自己的元文件（审计报告/水位/排除清单/索引/备份/视图），这些绝不能当记忆索引。"""
    n = (name or "").lower()
    if n.startswith("审计报告_"):
        return True
    if n in ("llm_organize_state.txt", "compacted_sources.txt", "index.json", "index_manifest.json"):
        return True
    if n.startswith("index.json.") or n.startswith("index_manifest"):
        return True
    if n.startswith("memory.db") or ".swap-" in n:
        return True
    if n.endswith(".html"):
        return True
    return False


def _is_excluded(rel, ex):
    """排除判定：精确路径或目录前缀（以 / 结尾），且在任意层级生效。

    例：填 "L1/" 时，L1/x.txt、facts/L1/x.txt、Memory/L1/x.txt 都会被排除——
    否则副记忆根（facts/、Memory/）下的同名目录永远排不掉。"""
    r = (rel or "").replace("\\", "/").strip("/")
    if not r:
        return False
    hay = "/" + r + "/"
    for e in (ex or []):
        e2 = str(e).replace("\\", "/").strip("/")
        if not e2:
            continue
        if e2.endswith("/"):
            e2 = e2.strip("/") + "/"
            if hay.startswith("/" + e2) or ("/" + e2) in hay:
                return True
        elif "/" + e2 + "/" in hay:
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
        except sqlite3.OperationalError as e:
            # 已升级过的库报 duplicate column 属正常；其余失败要能定位
            if "duplicate column" not in str(e).lower():
                _warn("upgrade edges.type column: %s" % db_path, e)
        try:
            c.execute("ALTER TABLE entries ADD COLUMN text_hash TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                _warn("upgrade entries.text_hash column: %s" % db_path, e)
        try:
            c.execute("CREATE INDEX IF NOT EXISTS ix_entries_hash ON entries(text_hash)")
        except sqlite3.Error as e:
            _warn("create entries hash index: %s" % db_path, e)
        _db_prepare_call_events(c)
        _db_sync_call_aggregates(c)
        c.commit()
        if upgraded:
            try:
                _db_rebuild_links(c)   # 旧库升级：重算一遍带类型的边
                c.commit()
            except sqlite3.Error as e:
                _warn("rebuild links after upgrade: %s" % db_path, e)
        _DB.update({"ok": True, "path": db_path, "conn": c})
        return c
    except Exception as e:
        # 打不开库时全部调用方都按「统计/索引不可用」降级，契约不变；
        # 但必须留痕，否则表现为「没有记忆」而不是「库坏了」
        _warn("open plugin database: %s" % db_path, e)
        _DB.update({"ok": False, "path": db_path, "conn": None})
        return None


def _db_prepare_call_events(c):
    """Keep only confirmed delivery events in the current statistics table.

    Older builds wrote an event when a candidate was selected, before the
    caller received it. Preserve that data for diagnosis, but do not expose it
    as successful calls.
    """
    rows = c.execute("PRAGMA table_info(call_events)").fetchall()
    primary_key = [r[1] for r in sorted((r for r in rows if r[5]), key=lambda r: r[5])]
    legacy_event_table = False
    if rows and primary_key != ["logical_id", "fact_id"]:
        legacy = "call_events_legacy_unconfirmed"
        suffix = 2
        while c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (legacy,)).fetchone():
            legacy = "call_events_legacy_unconfirmed_%d" % suffix
            suffix += 1
        c.execute("DROP INDEX IF EXISTS ix_call_events_fact_time")
        c.execute("ALTER TABLE call_events RENAME TO " + legacy)
        legacy_event_table = True
    c.execute(
        "CREATE TABLE IF NOT EXISTS call_events("
        "logical_id TEXT NOT NULL,fact_id TEXT NOT NULL,fact_version TEXT NOT NULL,"
        "channel TEXT NOT NULL,called_at TEXT NOT NULL,PRIMARY KEY(logical_id,fact_id))")
    c.execute("CREATE INDEX IF NOT EXISTS ix_call_events_fact_time ON call_events(fact_id,called_at)")
    c.execute("CREATE TABLE IF NOT EXISTS call_stats_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    c.execute("CREATE TABLE IF NOT EXISTS call_stats_gaps(started_at TEXT PRIMARY KEY,ended_at TEXT NOT NULL)")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if legacy_event_table:
        # Old rows were candidate selections, not confirmed delivery. Start a
        # fresh, truthful observation window instead of retaining their age.
        c.execute("INSERT OR REPLACE INTO call_stats_meta(key,value) VALUES('started_at',?)", (now,))
    else:
        c.execute("INSERT OR IGNORE INTO call_stats_meta(key,value) VALUES('started_at',?)", (now,))
    if legacy_event_table:
        c.execute("INSERT OR REPLACE INTO call_stats_meta(key,value) VALUES('legacy_unconfirmed_events','true')")


def _entry_fact_reference(entry):
    fact_id = str(entry.get("id") or "")
    if not fact_id:
        return None
    version = str(entry.get("text_hash") or entry.get("_hash") or "")
    if not version:
        norm = re.sub(r"\s+", "", str(entry.get("text") or entry.get("body") or entry.get("summary") or ""))
        version = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
    return {"id": fact_id, "version": version}


def _fact_references(entries):
    return [ref for ref in (_entry_fact_reference(e) for e in entries) if ref]


def _is_fact_source(source):
    """Only plugin-owned facts/ entries are eligible for direct recall/statistics."""
    normalized = str(source or "").replace("\\", "/").lstrip("/")
    return normalized.lower().startswith("facts/")


def _db_for(outpath):
    """按索引输出路径定位同目录的 memory.db（每角色一份）。"""
    dbp = os.path.join(os.path.dirname(outpath or ""), "memory.db") if outpath else ""
    if not dbp:
        return None
    if _DB["ok"] and _DB["path"] == dbp and _DB["conn"] is not None:
        return _DB["conn"]
    # ★换库（多角色切换）先把旧连接关掉：否则句柄泄漏、文件被锁住删不掉
    cn = _DB.get("conn")
    if cn is not None:
        try:
            cn.close()
        except Exception as e:
            # 关闭旧连接失败会导致换库时句柄泄漏
            _warn("close previous database connection", e)
        _DB.update({"ok": False, "path": "", "conn": None})
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
    th = e.get("_hash")
    if not th:
        norm = re.sub(r"\s+", "", str(e.get("text") or ""))
        th = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
    return (
        e.get("id") or "", e.get("anchor") or "", e.get("source") or "",
        e.get("time") or "", e.get("date") or "", e.get("period") or "",
        e.get("hour"), (e.get("head") or "")[:60], e.get("summary") or "",
        e.get("text") or "", hay, th,
        "|".join(e.get("people") or []), "|".join(e.get("topics") or []),
        "|".join(e.get("things") or []),
        int(e.get("access_count") or 0), e.get("last_access") or "",
        1 if e.get("retired") else 0, e.get("retired_at") or "",
    )


def _db_sync_call_aggregates(c):
    """Keep legacy lifecycle columns independent from delivery statistics."""
    return None


def _db_upsert(c, e):
    try:
        c.execute(
            "INSERT OR REPLACE INTO entries(id,anchor,source,time,date,period,hour,head,summary,body,hay,text_hash,"
            "people,topics,things,access_count,last_access,retired,retired_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _entry_row(e))
        return True
    except sqlite3.Error as error:
        # 批量导入逐条容错，失败条目要能定位到具体卡 ID
        _warn("upsert entry: %s" % (e.get("id") or ""), error)
        return False


def _db_backfill_hash(c):
    """旧库升级：给没有 text_hash 的条目补上（去重上限依赖它）。"""
    try:
        rows = c.execute("SELECT id, body FROM entries WHERE text_hash IS NULL OR text_hash=''").fetchall()
        if not rows:
            return 0
        upd = []
        for r in rows:
            norm = re.sub(r"\s+", "", r["body"] or "")
            upd.append((hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16], r["id"]))
        c.executemany("UPDATE entries SET text_hash=? WHERE id=?", upd)
        c.commit()
        return len(upd)
    except sqlite3.Error as e:
        _warn("backfill text_hash", e)
        return 0


def _db_import_json(c, json_path):
    """首次迁移：把旧的 index.json 导入库（只导一次）。"""
    try:
        if not json_path or not os.path.isfile(json_path):
            return 0
        if c.execute("SELECT COUNT(*) FROM entries").fetchone()[0] > 0:
            return 0
        data = json.load(io.open(json_path, encoding="utf-8")) or []
    except (OSError, ValueError, sqlite3.Error) as e:
        # 首次迁移读不出旧 index.json：返回 0 条，但必须留痕
        _warn("import legacy index json: %s" % json_path, e)
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
    """重建互链（标签 + 正文实体，共现边）。建库时跑一次，成本可控。

    ★只统计未软删的条目：节点/边是图谱的检索入口，若把已软删的卡也计入，
      图谱搜索仍然能搜到"已经删掉"的人和事（实测：画像里没了、图谱里还在）。
    """
    try:
        c.execute("DELETE FROM nodes")
        c.execute("DELETE FROM edges")
        nodes, edges = {}, {}
        for r in c.execute("SELECT id,head,summary,body,people,topics,things FROM entries "
                           "WHERE COALESCE(retired,0)=0"):
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
    except sqlite3.Error as e:
        # 关系重建失败会让图谱停在旧边，原为静默 pass
        _warn("rebuild graph links", e)


def _db_ready(req):
    """取连接；库为空而旧 index.json 在就先迁移（兼容只给 index 的旧调用）。"""
    idx = req.get("index") or ""
    # ★把索引路径同步到当前库：否则图谱/迁移等内部 helper 会按旧路径开库，
    #   顺手把当前连接关掉（Cannot operate on a closed database）。
    if idx:
        _set_index(idx)
    c = _db_for(idx)
    if c is None:
        return None
    try:
        if c.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0:
            _db_import_json(c, idx)
    except sqlite3.Error as e:
        _warn("check entries count for legacy import: %s" % idx, e)
    return c


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
    """Date decay only; call totals and call timestamps must not affect ranking."""
    return _entry_weight(entry)


def _db_vec_get(c, eid):
    """向量只缓存在进程内：落库会让 5 万条 × 2KB 膨胀 100MB+，而重算是毫秒级（批量）。"""
    return _VEC_CACHE.get("v:" + str(eid))


def _db_vec_put(c, eid, vec):
    try:
        if vec:
            _VEC_CACHE["v:" + str(eid)] = vec
            return
            c.execute("INSERT OR REPLACE INTO vec(id,data) VALUES(?,?)",
                      (eid, struct.pack("<%df" % len(vec), *vec)))
    except sqlite3.Error as e:
        _warn("persist entry vector: %s" % eid, e)


def db_load_all(outpath):
    """把库内容整份取出（兼容仍按 _load_index() 工作的旧算法，如 compactscan/audit）。"""
    c = _db_for(outpath)
    if c is None:
        return None
    try:
        return [_row_entry(c, r) for r in c.execute("SELECT * FROM entries")]
    except sqlite3.Error as e:
        _warn("load all entries: %s" % outpath, e)
        return None
