"""图谱构建、邻接扩展与图谱只读数据。"""

import os
import re
import urllib
import sqlite3


# >>> WP1 导入（自动生成，勿手改）
import dbside
import indexstore
from stat_tracker import _fact_call_stats
from state import _GRAPH, _GRAPH_CACHE, _warn
from textsim import extract_entities
# <<< WP1 导入

def build_graph():
    """从 index 抽节点（人/事/物/高频实词）与共现边。
    节点=出现在条目里的标签词；边=同一条目内两节点共现（权重=共现次数）。
    """
    nodes = {}
    edges = {}
    for e in indexstore._load_index():
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


def _graph_int(value, default, lo, hi):
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(lo, min(hi, n))


def _graph_like(text):
    """LIKE 预过滤用的转义；真正的标签匹配在 Python 里按 | 分割后精确比较。"""
    return str(text or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _graph_source(raw):
    """来源路径只作展示/节点 id，一律当不可信数据；统一成 / 并限长。"""
    return str(raw or "").replace("\\", "/")[:512]


def _graph_label(value, fallback):
    s = re.sub(r"\s+", " ", str(value or "")).strip()
    return (s[:80] or fallback)


def _graph_tag_where(name):
    """标签/正文匹配的 WHERE 子句；分页查询与计数必须共用同一份条件。"""
    tag_pat = "%|" + _graph_like(name) + "|%"
    text_pat = "%" + _graph_like(name) + "%"
    sql = ("retired=0 AND (('|' || people || '|') LIKE ? ESCAPE '\\' "
           "OR ('|' || topics || '|') LIKE ? ESCAPE '\\' "
           "OR ('|' || things || '|') LIKE ? ESCAPE '\\' "
           "OR head LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' OR body LIKE ? ESCAPE '\\')")
    return sql, (tag_pat, tag_pat, tag_pat, text_pat, text_pat, text_pat)


def _graph_tag_count(c, name):
    sql, args = _graph_tag_where(name)
    try:
        return int(c.execute("SELECT COUNT(*) FROM entries WHERE " + sql, args).fetchone()[0] or 0)
    except sqlite3.Error as e:
        _warn("count graph tag rows: %s" % name, e)
        return 0


def _graph_tag_rows(c, name, limit=80, offset=0):
    """按完整标签或正文实体找条目，并在 Python 侧精确过滤候选。

    ★分页要稳定：除了 time DESC 之外必须再按 id 排序，否则同一天的多条
      在两次查询里的相对顺序可能不同（真实库大量同秒写入），翻页会漏。
    """
    sql, args = _graph_tag_where(name)
    try:
        rows = c.execute(
            "SELECT id,source,head,summary,body,date,people,topics,things,access_count,last_access FROM entries "
            "WHERE " + sql + " ORDER BY time DESC, id ASC LIMIT ? OFFSET ?",
            args + (max(1, int(limit)), max(0, int(offset)))).fetchall()
    except sqlite3.Error as e:
        _warn("query graph tag rows: %s" % name, e)
        return []
    out = []
    for row in rows:
        tags = set((row["people"] or "").split("|") + (row["topics"] or "").split("|") + (row["things"] or "").split("|"))
        haystack = "\n".join((row["head"] or "", row["summary"] or "", row["body"] or "")).lower()
        if name in tags:
            out.append(row)
            continue
        low_name = str(name or "").lower()
        if re.search(r"[\u4e00-\u9fff]", low_name):
            matched = low_name in haystack
        else:
            matched = bool(re.search(r"(?<![a-z0-9_.-])" + re.escape(low_name) + r"(?![a-z0-9_.-])", haystack))
        if matched:
            out.append(row)
    return out


def _graph_call_stats(c, fact_id, fallback_last=""):
    """读取由交付确认事件生成的事实卡统计。"""
    stats = _fact_call_stats(c, fact_id)
    stats["last_called_at"] = stats.get("last_called_at") or ""
    return stats


def db_graphdata(req):
    """为 UI 返回有界关系图；graph 读连接仅 read-only 打开既有 memory.db。"""
    readonly = bool(req.get("readonly"))
    idx = req.get("index") or ""
    test_conn = globals().get("_graph_test_connection")
    if test_conn is not None:
        c = test_conn
    elif not readonly:
        c = dbside._db_ready(req)
    else:
        dbp = os.path.join(os.path.dirname(idx), "memory.db") if idx else ""
        if not dbp or not os.path.isfile(dbp):
            return None
        try:
            cached = _GRAPH_CACHE.get("conn")
            if cached is None or _GRAPH_CACHE.get("path") != dbp:
                if cached is not None:
                    try: cached.close()
                    except Exception as e: _warn("close readonly graph cache: %s" % dbp, e)
                uri = "file:" + urllib.parse.quote(os.path.abspath(dbp).replace("\\", "/"), safe="/:\\") + "?mode=ro"
                cached = sqlite3.connect(uri, uri=True, timeout=10)
                cached.row_factory = sqlite3.Row
                cached.execute("PRAGMA query_only=ON")
                # Fail closed unless this is an already initialized plugin DB.
                required = {r[0] for r in cached.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {"entries", "nodes", "edges"}.issubset(required):
                    cached.close()
                    return {"ok": False, "error": "memory.db schema is too old; rebuild the graph index"}
                cols = {r[1] for r in cached.execute("PRAGMA table_info(edges)")}
                if "type" not in cols:
                    cached.close()
                    return {"ok": False, "error": "memory.db schema is too old; rebuild the graph index"}
                _GRAPH_CACHE.update({"path": dbp, "conn": cached})
            c = cached
        except sqlite3.Error as e:
            _warn("open readonly graph database: %s" % dbp, e)
            return None
    if isinstance(c, dict) and c.get("ok") is False:
        return c
    if c is None:
        return None
    query = str(req.get("query") or "").strip().lower()[:200]
    expand_id = str(req.get("expand") or "").strip()[:512]
    top = _graph_int(req.get("top"), 8, 1, 200)
    max_nodes = _graph_int(req.get("max_nodes"), 100, 10, 250)
    # 关系分页：offset 是已返回的关系条数，relation_limit 是单页上限（默认 30、上限 200）。
    # 所有受限路径（单节点关系、来源展开、关联条目、全图概览）共用同一套语义，
    # 排序键固定为 (w DESC, name ASC)，同权重跨页顺序稳定，翻页不重不漏。
    relation_limit = _graph_int(req.get("relation_limit"), 30, 1, 200)
    offset = _graph_int(req.get("offset"), 0, 0, 1000000)
    entry_limit = _graph_int(req.get("entry_limit"), 24, 1, 200)
    entry_offset = _graph_int(req.get("entry_offset"), 0, 0, 1000000)
    max_edges = max(40, max_nodes * 4)
    include_entries = bool(req.get("include_entries"))
    nodes = {}
    edges = []
    truncated = False
    total_relations = 0
    returned = 0
    total_entries = 0
    entries_returned = 0

    def add_node(nid, label, kind, **extra):
        nonlocal truncated
        if nid in nodes:
            for k, v in extra.items():
                if v not in (None, ""):
                    nodes[nid][k] = v
            return nodes[nid]
        if len(nodes) >= max_nodes:
            truncated = True
            return None
        node = {"id": nid, "label": _graph_label(label, nid), "kind": kind}
        node.update(extra)
        nodes[nid] = node
        return node

    def add_edge(src, tgt, weight=1, etype="共现", kind="cooccurrence"):
        nonlocal truncated
        if src not in nodes or tgt not in nodes:
            return False
        if len(edges) >= max_edges:
            truncated = True
            return False
        try:
            w = max(1, int(weight or 1))
        except (TypeError, ValueError):
            w = 1
        edges.append({"source": src, "target": tgt, "weight": w, "type": etype, "kind": kind})
        return True

    def add_entry(row, via=None, relation="提及条目"):
        eid = "entry:" + str(row["id"])
        source = _graph_source(row["source"])
        call_stats = _graph_call_stats(c, row["id"], row["last_access"])
        if add_node(eid, row["head"] or row["summary"] or row["id"] or "记忆条目", "entry",
                    source=source, date=str(row["date"] or "")[:20], summary=str(row["summary"] or "")[:300],
                    **call_stats) is None:
            return None
        if via:
            add_edge(via, eid, 1, relation, "hierarchy")
        if source:
            sid = "source:" + source
            add_node(sid, source.rsplit("/", 1)[-1], "source", source=source)
            if sid in nodes:
                add_edge(sid, eid, 1, "包含条目", "hierarchy")
        return nodes.get(eid)

    if expand_id:
        # 展开目标不要求先在当前图里：页面点“展开”时可能不在概览节点集中。
        if expand_id.startswith("source:"):
            source = _graph_source(expand_id[len("source:"):])
            add_node(expand_id, source.rsplit("/", 1)[-1] or source or "来源未知", "source", source=source)
            if source:
                where = "retired=0 AND (source=? OR source=?)"
                args = (source, source.replace("/", "\\"))
                total_relations = int(c.execute(
                    "SELECT COUNT(*) FROM entries WHERE " + where, args).fetchone()[0] or 0)
                rows = c.execute(
                    "SELECT id,source,head,summary,date,access_count,last_access FROM entries "
                    "WHERE " + where + " ORDER BY time DESC, id ASC LIMIT ? OFFSET ?",
                    args + (relation_limit, offset)).fetchall()
                for row in rows:
                    # 只有真正交付出去的条目才消耗游标；节点预算不足时停在原处，
                    # 下一页从这条重新开始，绝不出现"跳过去但没给"的关系。
                    if add_entry(row, via=expand_id, relation="包含条目") is None:
                        truncated = True
                        break
                    returned += 1
        elif expand_id.startswith("entry:"):
            row = c.execute("SELECT id,source,head,summary,date,access_count,last_access FROM entries WHERE id=? AND retired=0",
                            (expand_id[len("entry:"):],)).fetchone()
            if row:
                add_entry(row)
                total_relations = returned = 1
        else:
            add_node(expand_id, expand_id, "seed")
            # 游标 = (w DESC, 对端名称 ASC)：同权重的关系严格按名称定序，跨页稳定。
            total_relations = int(c.execute(
                "SELECT COUNT(*) FROM edges WHERE a=? OR b=?", (expand_id, expand_id)).fetchone()[0] or 0)
            rows = c.execute(
                "SELECT a,b,w,type FROM edges WHERE a=? OR b=? "
                "ORDER BY w DESC, CASE WHEN a=? THEN b ELSE a END ASC, b ASC LIMIT ? OFFSET ?",
                (expand_id, expand_id, expand_id, relation_limit, offset)).fetchall()
            for r in rows:
                left = add_node(r["a"], r["a"], "neighbor")
                right = add_node(r["b"], r["b"], "neighbor")
                if left is None or right is None:
                    truncated = True
                    break
                if not add_edge(r["a"], r["b"], r["w"], (r["type"] or "共现").split("|")[0]):
                    truncated = True
                    break
                returned += 1
            # 关联条目是独立的一组关系，用自己的游标分页（默认每页 24 条）。
            total_entries = _graph_tag_count(c, expand_id)
            for row in _graph_tag_rows(c, expand_id, entry_limit, entry_offset):
                if add_entry(row, via=expand_id, relation="提及条目") is None:
                    truncated = True
                    break
                entries_returned += 1
    elif query:
        # 先精确、再前缀、最后子串，避免 “Alice” 被更热门的 “Malice” 抢走。
        pat = "%" + _graph_like(query) + "%"
        matched = []
        for r in c.execute("SELECT name,n FROM nodes WHERE name LIKE ? ESCAPE '\\' ORDER BY n DESC LIMIT 200", (pat,)).fetchall():
            name = str(r["name"] or "")
            low = name.lower()
            if not name or query not in low:
                continue
            rank = 0 if low == query else (1 if low.startswith(query) else 2)
            try:
                count = int(r["n"] or 0)
            except (TypeError, ValueError):
                count = 0
            matched.append((rank, -count, name, count))
        matched.sort()
        seeds = [(name, count) for _rank, _neg, name, count in matched[:3]]
        # 有精确同名节点时，条目只跟精确节点关联；否则 “血压” 会带出 “血压计”，
        # 而 “Alice” 不该把 “Malice” 的条目也算进来。
        exact = [name for rank, _neg, name, _count in matched if rank == 0][:3]
        entry_seeds = exact if exact else [name for name, _count in seeds]
        for name, count in seeds:
            add_node(name, name, "seed", count=count)
        # 搜索路径按种子逐个分页：offset 对每个种子各自生效，分页边界与单节点一致。
        # 没显式给 relation_limit 时沿用旧的 top（默认 8），保持既有行为。
        seed_page = relation_limit if req.get("relation_limit") is not None else top
        seed_total = seed_returned = 0
        for name, _count in seeds:
            degree = int(c.execute(
                "SELECT COUNT(*) FROM edges WHERE a=? OR b=?", (name, name)).fetchone()[0] or 0)
            seed_total = max(seed_total, degree)
            rows = c.execute(
                "SELECT a,b,w,type FROM edges WHERE a=? OR b=? "
                "ORDER BY w DESC, CASE WHEN a=? THEN b ELSE a END ASC, b ASC LIMIT ? OFFSET ?",
                (name, name, name, seed_page, offset)).fetchall()
            delivered = 0
            for r in rows:
                left = add_node(r["a"], r["a"], "neighbor")
                right = add_node(r["b"], r["b"], "neighbor")
                if left is None or right is None:
                    truncated = True
                    break
                if not add_edge(r["a"], r["b"], r["w"], (r["type"] or "共现").split("|")[0]):
                    truncated = True
                    break
                delivered += 1
            seed_returned = max(seed_returned, delivered)
        total_relations, returned = seed_total, seed_returned
        if include_entries:
            for name in entry_seeds:
                total_entries = max(total_entries, _graph_tag_count(c, name))
                entry_rows = _graph_tag_rows(c, name, entry_limit, entry_offset)
                delivered = 0
                for row in entry_rows:
                    if add_entry(row, via=name, relation="提及条目") is None:
                        truncated = True
                        break
                    delivered += 1
                entries_returned = max(entries_returned, delivered)
    else:
        # 概览：按同一游标分页，不再用 max_nodes*2 的隐式截断。
        total_relations = int(c.execute("SELECT COUNT(*) FROM edges").fetchone()[0] or 0)
        rows = c.execute(
            "SELECT a,b,w,type FROM edges ORDER BY w DESC, a ASC, b ASC LIMIT ? OFFSET ?",
            (relation_limit, offset)).fetchall()
        for r in rows:
            left = add_node(r["a"], r["a"], "node")
            right = add_node(r["b"], r["b"], "node")
            if left is None or right is None:
                truncated = True
                break
            if not add_edge(r["a"], r["b"], r["w"], (r["type"] or "共现").split("|")[0]):
                truncated = True
                break
            returned += 1

    # 同一对节点可能因实体/来源/多层展开被加多次；按无向边去重。
    seen = set()
    deduped = []
    for e in edges:
        key = (tuple(sorted((e["source"], e["target"]))), e["type"], e["kind"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    # 软删卡只从图谱里隐去，不删除；把数量报给页面，用户能确认"确实被排除"，
    # 而不是以为节点还在。
    try:
        retired_hidden = int(c.execute(
            "SELECT COUNT(*) FROM entries WHERE COALESCE(retired,0)=1 "
            "AND lower(replace(source,char(92),'/')) LIKE 'facts/%'").fetchone()[0] or 0)
    except sqlite3.Error:
        retired_hidden = 0
    has_more = (offset + returned) < total_relations
    return {
        "nodes": list(nodes.values()), "edges": deduped, "query": query,
        # truncated 保留旧语义（有截断就要提示用户），分页信息单独给出
        "truncated": bool(truncated or has_more),
        "returned": returned,
        "offset": offset,
        "next_offset": offset + returned,
        "total_relations": total_relations,
        "has_more": has_more,
        "entries_returned": entries_returned,
        "entry_offset": entry_offset,
        "total_entries": total_entries,
        "entries_has_more": (entry_offset + entries_returned) < total_entries,
        "retired_hidden": retired_hidden,
    }
