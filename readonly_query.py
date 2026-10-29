"""只读浏览、画像查询、统计与关系查询。"""

import os
import io
import json
import datetime
import urllib
import sqlite3


# >>> WP1 导入（自动生成，勿手改）
import dbside
from dbside import _db_where_time, _is_fact_source, _row_entry
from graph_indexer import _graph_int
import indexstore
from stat_tracker import _fact_call_stats
from state import _PROFILE_KIND_COLUMNS, _warn
from textsim import rel_time
# <<< WP1 导入

def browse(query=None, who=None, topic=None, thing=None, date_from=None, date_to=None, offset=0, limit=50,
           facts_only=False, _entries=None, usage=None, importance=None, sort="time"):
    """给内嵌面板用：按关键词/人/事/物/时间翻条目，返回结构化列表 + 可直接显示的文本。"""
    source_data = indexstore._load_index() if _entries is None else _entries
    data = [e for e in source_data if not e.get("retired") and
            (not facts_only or _is_fact_source(e.get("source")))]
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
        known_importance = e.get("importance") is not None and str(e.get("importance")).strip() != ""
        if importance == "known" and not known_importance:
            return False
        if importance == "unset" and known_importance:
            return False
        return True

    hits = [e for e in data if _ok(e)]
    if sort == "call_count":
        hits.sort(key=lambda e: (int(e.get("call_count") or 0), str(e.get("time") or "")), reverse=True)
    elif sort == "calls_7d":
        hits.sort(key=lambda e: (int(e.get("calls_7d") or 0), str(e.get("time") or "")), reverse=True)
    elif sort == "importance":
        hits.sort(key=lambda e: (str(e.get("importance") or "").lower(), str(e.get("time") or "")))
    elif sort == "name":
        hits.sort(key=lambda e: str(e.get("head") or e.get("summary") or e.get("id") or "").lower())
    else:
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
            "importance": e.get("importance"),
            "importance_source": e.get("importance_source") or "未知",
        })
        if facts_only:
            items[-1].update({
                "statistics_available": False, "call_count": None,
                "automatic_call_count": None, "tool_call_count": None,
                "calls_7d": None, "calls_7d_observed": None,
                "statistics_complete": False, "statistics_start_time": "",
                "statistics_gap_count_7d": 0, "statistics_gap_open": False,
                "last_called_at": "",
            })
    lines = []
    for it in items:
        label = "、".join(list(it["topics"]) + list(it["things"])) or it["head"] or "记忆"
        lines.append("- [%s] %s：%s（出处 %s）" % (it["date"], label, it["summary"], it["source"]))
    return {"total": total, "offset": off, "items": items, "text": "\n".join(lines)}


def db_browse(req):
    c = dbside._db_ready(req)
    if c is None:
        return None
    return _db_browse_on_connection(c, req)


def _db_browse_on_connection(c, req):
    where, args = _db_where_time(req)
    has_call_events = bool(c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='call_events'"
    ).fetchone())
    columns = {r[1] for r in c.execute("PRAGMA table_info(entries)").fetchall()}
    if req.get("facts_only"):
        where.append("lower(replace(source,char(92),'/')) LIKE 'facts/%'")
    usage = (req.get("usage") or "").strip().lower()
    if usage in ("used", "unused"):
        if not has_call_events:
            raise sqlite3.OperationalError("call statistics are unavailable")
        exists = "EXISTS" if usage == "used" else "NOT EXISTS"
        where.append(exists + " (SELECT 1 FROM call_events ce WHERE ce.fact_id=entries.id)")
    importance = (req.get("importance") or "").strip().lower()
    if importance in ("known", "unset"):
        if "importance" not in columns:
            where.append("0" if importance == "known" else "1")
        elif importance == "known":
            where.append("importance IS NOT NULL AND TRIM(CAST(importance AS TEXT))!=''")
        else:
            where.append("(importance IS NULL OR TRIM(CAST(importance AS TEXT))='')")
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
    sort = (req.get("sort") or "name").strip().lower()
    if sort in ("call_count", "calls_7d") and not has_call_events:
        raise sqlite3.OperationalError("call statistics are unavailable")
    if sort == "call_count" and has_call_events:
        order_sql = "COALESCE((SELECT COUNT(*) FROM call_events ce WHERE ce.fact_id=entries.id),0) DESC, time DESC"
    elif sort == "calls_7d" and has_call_events:
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)).isoformat()
        order_sql = "(SELECT COUNT(*) FROM call_events ce WHERE ce.fact_id=entries.id AND ce.called_at>?) DESC, time DESC"
        args.append(cutoff)
    elif sort == "importance" and "importance" in columns:
        order_sql = "importance COLLATE NOCASE ASC, time DESC"
    elif sort == "last_called":
        # ★排序必须用"已确认调用"的数据源：成功回执写的是 call_events，不写 entries.last_access
        if has_call_events:
            last_call = ("(SELECT MAX(ce.called_at) FROM call_events ce WHERE ce.fact_id=entries.id)")
            order_sql = (f"CASE WHEN COALESCE({last_call},'') = '' THEN 1 ELSE 0 END ASC, "
                         f"{last_call} DESC, time DESC")
        else:
            order_sql = ("CASE WHEN COALESCE(last_access,'') = '' THEN 1 ELSE 0 END ASC, "
                         "last_access DESC, time DESC")
    else:
        order_sql = "head COLLATE NOCASE ASC, time DESC"
    rows = c.execute("SELECT * FROM entries WHERE " + wsql + " ORDER BY " + order_sql + " LIMIT ? OFFSET ?",
                     args + [lim, off]).fetchall()
    items, lines = [], []
    gap_path = os.path.join(os.path.dirname(req.get("index") or ""), "call_stats_gap.txt")
    gap_open_since = ""
    try:
        raw_gap = io.open(gap_path, encoding="utf-8").read().strip()
        gap_dt = datetime.datetime.fromisoformat(raw_gap.replace("Z", "+00:00"))
        if gap_dt.tzinfo is None:
            gap_dt = gap_dt.replace(tzinfo=datetime.timezone.utc)
        gap_open_since = gap_dt.astimezone(datetime.timezone.utc).isoformat()
        if c.execute("SELECT 1 FROM call_stats_gaps WHERE started_at=?", (gap_open_since,)).fetchone():
            gap_open_since = ""  # already closed; marker removal may have been interrupted
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        gap_open_since = "invalid"
    for r in rows:
        d = _row_entry(c, r)
        d["rel_time"] = rel_time(d.get("date", ""))
        # 管理页只读展示生命周期统计；这些字段来自插件自己的 memory.db，
        # 不把候选浏览或失败请求误算成调用。
        call_stats = _fact_call_stats(c, d.get("id"), gap_open_since=gap_open_since)
        item = {k2: d.get(k2) for k2 in (
            "id", "date", "period", "head", "summary", "source", "people", "topics", "things",
            "rel_time", "access_count", "last_access",
        )}
        # 版本号：人工编辑/软删提交时用来做乐观并发校验（entries.text_hash）
        item["version"] = str(d.get("text_hash") or "")
        # 正文：管理表右键编辑要预填当前内容，不给就只能让用户重打一遍
        body_text = str(d.get("text") or "")
        item["excerpt"] = body_text[:1600]
        item["excerpt_truncated"] = len(body_text) > 1600
        # 本版本事实条目契约及侧车 schema 都没有可核实的重要度字段。
        item["importance"] = None
        item["importance_source"] = "未知"
        if "importance" in d:
            item["importance"] = d.get("importance")
        if "importance_source" in d:
            item["importance_source"] = d.get("importance_source")
        item.update(call_stats)
        items.append(item)
        label = "、".join((d.get("topics") or []) + (d.get("things") or [])) or (d.get("head") or "记忆")
        lines.append("- [%s] %s：%s（出处 %s）" % (d.get("date"), label, d.get("summary"), d.get("source")))
    return {"total": total, "offset": off, "items": items, "text": "\n".join(lines)}


def db_browse_readonly(req):
    """Read an existing sidecar DB for the management page without schema writes."""
    idx = req.get("index") or ""
    dbp = os.path.join(os.path.dirname(idx), "memory.db") if idx else ""
    if not dbp or not os.path.isfile(dbp):
        return None
    try:
        uri = "file:" + urllib.parse.quote(os.path.abspath(dbp).replace("\\", "/"), safe="/:\\") + "?mode=ro"
        c = sqlite3.connect(uri, uri=True, timeout=5)
        c.row_factory = sqlite3.Row
        try:
            if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='entries'").fetchone():
                return None
            return _db_browse_on_connection(c, req)
        finally:
            c.close()
    except Exception as e:
        _warn('open readonly database for browse', e)
        return None


def _profile_kind_items(entry):
    items = []
    for kind, column in _PROFILE_KIND_COLUMNS.items():
        for name in entry.get(column) or []:
            clean = str(name or "").strip()
            if clean:
                items.append({"kind": kind, "name": clean})
    return items


def db_entities_readonly(req):
    """列出现有事实卡里出现过的标签（默认人物），供画像页直接点选。

    实测反馈：画像只靠搜索不够直观。这里把"事实卡里真实出现过的标签"列出来，
    用户点一下就能看画像，不必先知道名字。
    """
    idx = req.get("index") or ""
    dbp = os.path.join(os.path.dirname(idx), "memory.db") if idx else ""
    if not dbp or not os.path.isfile(dbp):
        return None
    kind = str(req.get("kind") or "person").strip().lower()
    column = _PROFILE_KIND_COLUMNS.get(kind)
    if column is None:
        return {"invalid": True, "error": "a valid entity kind is required"}
    try:
        limit = max(1, min(200, int(req.get("limit") or 60)))
    except (TypeError, ValueError):
        limit = 60
    counts = {}
    try:
        uri = "file:" + urllib.parse.quote(os.path.abspath(dbp).replace("\\", "/"), safe="/:\\") + "?mode=ro"
        c = sqlite3.connect(uri, uri=True, timeout=5)
        c.row_factory = sqlite3.Row
        try:
            for row in c.execute(
                    "SELECT " + column + " AS tag FROM entries "
                    "WHERE COALESCE(retired,0)=0 "
                    "AND lower(replace(source,char(92),'/')) LIKE 'facts/%'"):
                for name in str(row["tag"] or "").split("|"):
                    name = name.strip()
                    if name:
                        counts[name] = counts.get(name, 0) + 1
        finally:
            c.close()
    except sqlite3.Error as e:
        _warn("list entities: %s" % dbp, e)
        return None
    items = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    return {
        "kind": kind,
        "items": [{"name": name, "count": count} for name, count in items],
        "total": len(counts),
        "truncated": len(counts) > limit,
    }


def db_profile_readonly(req):
    """Build a read-only, fact-backed person/event/object portrait."""
    idx = req.get("index") or ""
    dbp = os.path.join(os.path.dirname(idx), "memory.db") if idx else ""
    if not dbp or not os.path.isfile(dbp):
        return None
    name = str(req.get("name") or "").strip()[:120]
    requested_kind = str(req.get("kind") or "auto").strip().lower()
    if not name or requested_kind not in ("auto",) + tuple(_PROFILE_KIND_COLUMNS):
        return {"invalid": True, "error": "a valid entity name and kind are required"}
    try:
        uri = "file:" + urllib.parse.quote(os.path.abspath(dbp).replace("\\", "/"), safe="/:\\") + "?mode=ro"
        c = sqlite3.connect(uri, uri=True, timeout=5)
        c.row_factory = sqlite3.Row
        try:
            tables = {row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "entries" not in tables:
                return None
            columns = {row[1] for row in c.execute("PRAGMA table_info(entries)")}
            required = {"id", "source", "date", "time", "head", "summary", "people", "topics", "things", "retired"}
            if not required.issubset(columns):
                return None
            candidates = c.execute(
                "SELECT * FROM entries WHERE retired=0 "
                "AND lower(replace(source,char(92),'/')) LIKE 'facts/%' "
                "AND (instr('|' || COALESCE(people,'') || '|','|' || ? || '|')>0 "
                "OR instr('|' || COALESCE(topics,'') || '|','|' || ? || '|')>0 "
                "OR instr('|' || COALESCE(things,'') || '|','|' || ? || '|')>0) "
                "ORDER BY date DESC,time DESC,id",
                (name, name, name),
            ).fetchall()
            matched_by_kind = {kind: [] for kind in _PROFILE_KIND_COLUMNS}
            for row in candidates:
                entry = _row_entry(c, row)
                for kind, column in _PROFILE_KIND_COLUMNS.items():
                    if name in entry.get(column, []):
                        matched_by_kind[kind].append((row, entry))
            available_kinds = [kind for kind, rows in matched_by_kind.items() if rows]
            if requested_kind == "auto":
                if len(available_kinds) != 1:
                    return {
                        "entity": {"name": name, "kind": "", "basis": "事实卡标签"},
                        "available_kinds": available_kinds,
                        "requires_kind": len(available_kinds) > 1,
                        "found": False,
                        "total": 0,
                        "items": [],
                        "relations": [],
                    }
                kind = available_kinds[0]
            else:
                kind = requested_kind
            matched = matched_by_kind.get(kind, [])
            if not matched:
                return {
                    "entity": {"name": name, "kind": kind, "basis": "事实卡标签"},
                    "available_kinds": available_kinds,
                    "requires_kind": False,
                    "found": False,
                    "total": 0,
                    "items": [],
                    "relations": [],
                }

            relation_counts = {}
            relation_evidence = {}
            for row, entry in matched:
                fact_id = str(entry.get("id") or "")
                source = str(entry.get("source") or "")
                for peer in _profile_kind_items(entry):
                    if peer["kind"] == kind and peer["name"] == name:
                        continue
                    key = (peer["kind"], peer["name"])
                    relation_counts[key] = relation_counts.get(key, 0) + 1
                    evidence = relation_evidence.setdefault(key, [])
                    if len(evidence) < 8:
                        evidence.append({"fact_id": fact_id, "source": source})
            relations = [
                {
                    "id": "%s:%s" % (peer_kind, peer_name),
                    "kind": peer_kind,
                    "name": peer_name,
                    "evidence_count": count,
                    "evidence": relation_evidence[(peer_kind, peer_name)],
                }
                for (peer_kind, peer_name), count in relation_counts.items()
            ]
            relations.sort(key=lambda item: (-item["evidence_count"], item["kind"], item["name"].casefold()))

            offset = _graph_int(req.get("offset"), 0, 0, 100000)
            limit = _graph_int(req.get("limit"), 50, 1, 100)
            gap_open_since = ""
            gap_path = os.path.join(os.path.dirname(idx), "call_stats_gap.txt")
            try:
                raw_gap = io.open(gap_path, encoding="utf-8").read().strip()
                gap_dt = datetime.datetime.fromisoformat(raw_gap.replace("Z", "+00:00"))
                if gap_dt.tzinfo is None:
                    gap_dt = gap_dt.replace(tzinfo=datetime.timezone.utc)
                gap_open_since = gap_dt.astimezone(datetime.timezone.utc).isoformat()
                if "call_stats_gaps" in tables and c.execute(
                    "SELECT 1 FROM call_stats_gaps WHERE started_at=?", (gap_open_since,)
                ).fetchone():
                    gap_open_since = ""
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as e:
                gap_open_since = "invalid"
            # The profile list uses the same explicit statistics contract as
            # the fact-card management list.  Filtering and ordering happen
            # before paging so the UI can access the complete relation set.
            profile_stats = {}
            for row, entry in matched:
                profile_stats[str(entry.get("id") or "")] = _fact_call_stats(
                    c, entry.get("id"), gap_open_since=gap_open_since)
            profile_items = list(matched)
            usage = str(req.get("usage") or "").strip().lower()
            if usage in ("used", "unused"):
                profile_items = [
                    pair for pair in profile_items
                    # An unavailable statistics sidecar must not be presented
                    # as a zero-call card.  Only cards with a verifiable
                    # statistics record participate in the used/unused filter;
                    # callers can still see them under the unfiltered view,
                    # where the row is marked “统计不可用”.
                    if profile_stats.get(str(pair[1].get("id") or ""), {}).get("statistics_available") is True
                    and ((int(profile_stats.get(str(pair[1].get("id") or ""), {}).get("call_count") or 0) > 0)
                         == (usage == "used"))
                ]
            importance = str(req.get("importance") or "").strip().lower()
            if importance in ("known", "unset"):
                def _has_importance(pair):
                    value = pair[1].get("importance")
                    return value is not None and str(value).strip() != ""
                profile_items = [pair for pair in profile_items
                                 if _has_importance(pair) == (importance == "known")]
            sort = str(req.get("sort") or "name").strip().lower()
            if sort == "call_count":
                profile_items.sort(key=lambda pair: (
                    int(profile_stats.get(str(pair[1].get("id") or ""), {}).get("call_count") or 0),
                    str(pair[1].get("time") or "")), reverse=True)
            elif sort == "calls_7d":
                def _recent_calls(pair):
                    stats = profile_stats.get(str(pair[1].get("id") or ""), {})
                    # For a still-growing statistics window, the UI exposes
                    # calls_7d_observed.  Use that same observed value for
                    # ordering; treating a non-complete window as zero would
                    # put active cards behind newer cards with no calls.
                    value = stats.get("calls_7d")
                    if value is None:
                        value = stats.get("calls_7d_observed")
                    return int(value or 0)
                profile_items.sort(key=lambda pair: (
                    _recent_calls(pair),
                    str(pair[1].get("time") or "")), reverse=True)
            elif sort == "last_called":
                profile_items.sort(key=lambda pair: (
                    str(profile_stats.get(str(pair[1].get("id") or ""), {}).get("last_called_at") or ""),
                    str(pair[1].get("time") or "")), reverse=True)
            elif sort == "importance":
                profile_items.sort(key=lambda pair: (
                    str(pair[1].get("importance") or "").casefold(),
                    str(pair[1].get("time") or "")))
            else:
                profile_items.sort(key=lambda pair: (
                    str(pair[1].get("head") or pair[1].get("summary") or pair[1].get("id") or "").casefold(),
                    str(pair[1].get("time") or "")))
            page = profile_items[offset:offset + limit]
            items = []
            for row, entry in page:
                item = {
                    key: entry.get(key) for key in (
                        "id", "date", "period", "head", "summary", "source", "people", "topics", "things"
                    )
                }
                body_text = str(entry.get("text") or "")
                item["excerpt"] = body_text[:1600]
                item["excerpt_truncated"] = len(body_text) > 1600
                # 版本号：与 /facts 同一口径，供人工编辑/软删做并发校验
                item["version"] = str(entry.get("text_hash") or "")
                item["related_entities"] = _profile_kind_items(entry)
                item["importance"] = entry.get("importance") if "importance" in entry else None
                item["importance_source"] = entry.get("importance_source") if "importance_source" in entry else "未知"
                item_stats = profile_stats.get(item["id"])
                item.update(item_stats if item_stats is not None else _fact_call_stats(
                    c, item["id"], gap_open_since=gap_open_since))
                items.append(item)
            return {
                "entity": {"name": name, "kind": kind, "basis": "事实卡标签"},
                "available_kinds": available_kinds,
                "requires_kind": False,
                "found": True,
                "total": len(profile_items),
                "offset": offset,
                "limit": limit,
                "has_more": offset + len(items) < len(profile_items),
                "relations": relations,
                "relations_complete": True,
                "items": items,
                "text": "按精确标签关联 %d 张有效事实卡；实体关系表示共同出现在这些卡片中，不代表因果或身份消歧。" % len(profile_items),
            }
        finally:
            c.close()
    except Exception as e:
        _warn('open readonly database for profile', e)
        return None


def _browse_json_readonly(req):
    """Read only the explicitly configured index JSON as a safe DB fallback."""
    # Without call_events there is no honest way to determine unused cards.
    if (req.get("usage") or "").strip().lower() in ("used", "unused"):
        return None
    idx = req.get("index") or ""
    if not idx or not os.path.isfile(idx):
        return None
    try:
        with io.open(idx, encoding="utf-8") as f:
            entries = json.load(f)
        if not isinstance(entries, list):
            return None
    except Exception as e:
        _warn('read index json for readonly fallback', e)
        return None
    return browse(
        req.get("query"), req.get("who"), req.get("topic"), req.get("thing"),
        req.get("date_from") or None, req.get("date_to") or None,
        int(req.get("offset") or 0), int(req.get("limit") or 50),
        bool(req.get("facts_only")), _entries=entries,
        usage=req.get("usage"), importance=(req.get("importance") or "").strip().lower(),
        sort=(req.get("sort") or "name").strip().lower(),
    )


def db_browse_request(req):
    """Route management browsing without ever opening the sidecar DB for writes."""
    if req.get("readonly"):
        result = db_browse_readonly(req)
        if result is not None:
            return result
        if (req.get("usage") or "").strip().lower() in ("used", "unused"):
            return None
        return _browse_json_readonly(req)
    result = db_browse(req)
    if result is not None:
        return result
    return browse(
        req.get("query"), req.get("who"), req.get("topic"), req.get("thing"),
        req.get("date_from") or None, req.get("date_to") or None,
        int(req.get("offset") or 0), int(req.get("limit") or 50),
        bool(req.get("facts_only")), usage=req.get("usage"),
        importance=(req.get("importance") or "").strip().lower(),
        sort=(req.get("sort") or "name").strip().lower(),
    )


def db_stats(req):
    c = dbside._db_ready(req)
    if c is None:
        return None
    n = c.execute("SELECT COUNT(*) FROM entries WHERE retired=0").fetchone()[0]
    row = c.execute("SELECT MIN(date), MAX(date) FROM entries WHERE retired=0").fetchone()
    first, last = (row[0] or ""), (row[1] or "")
    return {"count": n, "first": first, "last": last,
            "text": "索引共 %d 条，覆盖 %s ~ %s" % (n, first or "?", last or "?")}


def db_links(req):
    c = dbside._db_ready(req)
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
