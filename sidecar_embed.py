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
import hashlib
import time
import urllib.parse
import uuid

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# 由 recall_engine.py 提供：_cos, _embed, _ensure_model, _entry_vector, _flush_life, _index_entries, _settle_dir, _touch, bytime, bywhen, compact_scan, db_bywhen, db_debug, db_recall, fetch, fetch_settled, find_model_dir, index_stats, maintain_index, open_loops, profile_card, recall, render_view, settle, to_bookmarks


# ============================================================
# 记忆召回（纯标准库，不依赖 torch）
# 打分逻辑与 recall_engine.py 同源：标签(人/事/物) > 标题 > 摘要 > 正文。
# ============================================================

# 由 state.py 提供：CJK_RUN_RE, DECAY_FLOOR, DECAY_HALF_LIFE_DAYS, HARD_DELETE_DAYS, PEOPLE, PERIOD_HOUR, SOFT_DELETE_DAYS, THINGS, TOPICS, WORD_RE, W_HEAD, W_SUMMARY, W_TAG, W_TEXT, _CACHE, _CACHE_MAX, _CACHE_ORDER, _CFG, _DB, _DB_SCHEMA, _ENTITY_PATTERNS, _ENTITY_STOP, _GRAPH, _GRAPH_CACHE, _HERE, _IDX, _LIFE, _MODEL, _NEG, _PROFILE_KIND_COLUMNS, _VEC_CACHE, _cache_put, _find_index, _set_index











# 由 indexstore.py 提供：_df, _idf, _load_index, _vocab, tokenize






# 由 textsim.py 提供：_audit_card_id, _decay_factor, _entry_weight, _extract_hour, _fact_files, _field_text, _first_sent, _guess_time, _jaccard, _looks_important, _parse_fact, _sim, _split_blocks, _split_fine, _tag, _tokens, extract_entities, parse_time_hint, rel_time, score_entry














# ===== 记忆生命周期：访问计数 + 强化（写回节流） =====







# ===== 记忆生命周期：衰减淡出（半衰期）+ 软删→物理清 =====












# ===== 图谱路：人/事/物 节点 + 共现边，顺关系链扩一层 =====



# 由 graph_indexer.py 提供：_graph_call_stats, _graph_int, _graph_label, _graph_like, _graph_source, _graph_tag_rows, _neighbors, build_graph, db_graphdata, graph_expand, links








# ===== 可视化：把索引导成一页 HTML（本地打开看） =====





# ============================================================
# 主循环
# ============================================================

# ============================================================
# 索引构建（从 tag_index.py 合并进来，内置，不依赖外部脚本）
# 这是"人/事/物"打标规则的唯一出处，改规则只改这里。
# ============================================================













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
        except (OSError, ValueError) as e:
            _warn('read audit state for build', e)
            old = {}
    entries = []
    skip_dirs = {"settled", "backup", "备份", "archive", "MemoryFacts"}
    ex = set(str(x).replace("\\", "/") for x in (exclude or []))
    jobs = [(memdir, "")]
    used_prefix = {""}
    for d in (extra_dirs or []):
        if d and os.path.isdir(d):
            # ★只有 MemoryFacts 这个目录才叫 facts/；其它记忆根用目录名做前缀。
            #   否则副记忆根也会被当成"我们自己的事实层"，压缩逻辑会永远跳过它们。
            base_name = os.path.basename(os.path.normpath(d))
            pref = "facts/" if base_name.lower() == "memoryfacts" else base_name + "/"
            if pref in used_prefix:
                i = 2
                while "%s%d/" % (base_name, i) in used_prefix:
                    i += 1
                pref = "%s%d/" % (base_name, i)
            used_prefix.add(pref)
            jobs.append((d, pref))
    allfiles = []
    for base, prefix in jobs:
        for root, dirs, _files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for fn in _files:
                if fn.lower().endswith((".md", ".txt")) and ".bak" not in fn.lower() and not _is_meta_file(fn):
                    allfiles.append((os.path.join(root, fn), base, prefix))
    for fpath, base, prefix in sorted(allfiles, key=lambda x: x[0]):
        fname = os.path.basename(fpath)
        inner = os.path.relpath(fpath, base).replace("\\", "/")
        rel = prefix + inner
        if _is_excluded(inner, ex) or _is_excluded(rel, ex):
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


















# ============================================================
# ① 自动压缩 / ② 审计 / ③ 写入合并 / ⑤ 实体图谱 / ⑥ 浏览
# ============================================================







# ---------- ① 自动压缩 ----------



# ---------- ② 审计 ----------







# 由 audit_ops.py 提供：_audit_consume_proposals, _audit_fact_records, _audit_issue_id, _audit_load_state, _audit_plugin_paths, _audit_restore_proposal_inputs, _audit_restore_state, _audit_save_state, _audit_sync_db, _audit_sync_rows, _audit_target_card_identity, audit_apply, audit_auto, audit_history, audit_scan, audit_undo, fact_merge, fact_propose, proposal_status






































# ---------- ③ 写入合并 ----------



# ---------- ⑤ 实体抽取（规则，不上模型） ----------






# ---------- ⑥ 内嵌浏览 ----------

# 由 readonly_query.py 提供：_browse_json_readonly, _db_browse_on_connection, _profile_kind_items, browse, db_browse, db_browse_readonly, db_browse_request, db_links, db_profile_readonly, db_stats
















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

# >>> WP1 导入（自动生成，勿手改）
import audit_ops
import fact_ops
from audit_ops import (
    _audit_load_state,
    audit_auto,
    audit_history,
    audit_scan,
    audit_undo,
    fact_merge,
    fact_propose,
    proposal_status,
)
import dbside
import impression
from dbside import (
    _db_backfill_hash,
    _db_import_json,
    _db_rebuild_links,
    _db_sync_call_aggregates,
    _db_upsert,
    _fact_references,
    _is_excluded,
    _is_meta_file,
)
from graph_indexer import build_graph, db_graphdata, links
import indexstore
from indexstore import _df, _vocab
import readonly_query
from readonly_query import browse, db_browse, db_browse_request, db_links, db_profile_readonly, db_stats
from recall_engine import (
    _embed,
    _ensure_model,
    bytime,
    bywhen,
    compact_scan,
    db_bywhen,
    db_debug,
    db_recall,
    fetch,
    index_stats,
    maintain_index,
    open_loops,
    profile_card,
    recall,
    render_view,
    settle,
    to_bookmarks,
)
from stat_tracker import db_record_calls
from state import (
    _CFG,
    _DB,
    _GRAPH,
    _GRAPH_CACHE,
    _HERE,
    _IDX,
    _MODEL,
    _VEC_CACHE,
    _find_index,
    _set_index,
    _warn,
)
from textsim import _extract_hour, _first_sent, _guess_time, _parse_fact, _split_blocks, _tag, parse_time_hint
from state import (
    _cache_put,
    _find_index,
    _set_index,
    _warn,
)
from textsim import (
    _audit_card_id,
    _decay_factor,
    _entry_weight,
    _extract_hour,
    _fact_files,
    _field_text,
    _first_sent,
    _guess_time,
    _jaccard,
    _looks_important,
    _parse_fact,
    _sim,
    _split_blocks,
    _split_fine,
    _tag,
    _tokens,
    extract_entities,
    parse_time_hint,
    rel_time,
    score_entry,
)
from dbside import (
    _cat,
    _db_backfill_hash,
    _db_boost,
    _db_for,
    _db_import_json,
    _db_open,
    _db_prepare_call_events,
    _db_ready,
    _db_rebuild_links,
    _db_score,
    _db_sync_call_aggregates,
    _db_upsert,
    _db_vec_get,
    _db_vec_put,
    _db_where_time,
    _edge_type,
    _entry_fact_reference,
    _entry_row,
    _fact_references,
    _is_excluded,
    _is_fact_source,
    _is_meta_file,
    _row_entry,
    db_load_all,
)
from indexstore import (
    _df,
    _idf,
    _load_index,
    _vocab,
    tokenize,
)
from stat_tracker import (
    _db_stats_start,
    _fact_call_stats,
    db_record_calls,
)
from graph_indexer import (
    _graph_call_stats,
    _graph_int,
    _graph_label,
    _graph_like,
    _graph_source,
    _graph_tag_rows,
    _neighbors,
    build_graph,
    db_graphdata,
    graph_expand,
    links,
)
from audit_ops import (
    _audit_consume_proposals,
    _audit_fact_records,
    _audit_issue_id,
    _audit_load_state,
    _audit_plugin_paths,
    _audit_restore_proposal_inputs,
    _audit_restore_state,
    _audit_save_state,
    _audit_sync_db,
    _audit_sync_rows,
    _audit_target_card_identity,
    audit_apply,
    audit_auto,
    audit_history,
    audit_scan,
    audit_undo,
    fact_merge,
    fact_propose,
    proposal_status,
)
from recall_engine import (
    _cos,
    _embed,
    _ensure_model,
    _entry_vector,
    _flush_life,
    _index_entries,
    _settle_dir,
    _touch,
    bytime,
    bywhen,
    compact_scan,
    db_bywhen,
    db_debug,
    db_recall,
    fetch,
    fetch_settled,
    find_model_dir,
    index_stats,
    maintain_index,
    open_loops,
    profile_card,
    recall,
    render_view,
    settle,
    to_bookmarks,
)
from readonly_query import (
    _browse_json_readonly,
    _db_browse_on_connection,
    _profile_kind_items,
    browse,
    db_browse,
    db_browse_readonly,
    db_browse_request,
    db_links,
    db_profile_readonly,
    db_stats,
)
from state import (
    CJK_RUN_RE,
    DECAY_FLOOR,
    DECAY_HALF_LIFE_DAYS,
    HARD_DELETE_DAYS,
    PEOPLE,
    PERIOD_HOUR,
    SOFT_DELETE_DAYS,
    THINGS,
    TOPICS,
    WORD_RE,
    W_HEAD,
    W_SUMMARY,
    W_TAG,
    W_TEXT,
    _CACHE,
    _CACHE_MAX,
    _CACHE_ORDER,
    _CFG,
    _DB,
    _DB_SCHEMA,
    _ENTITY_PATTERNS,
    _ENTITY_STOP,
    _GRAPH,
    _GRAPH_CACHE,
    _HERE,
    _IDX,
    _LIFE,
    _MODEL,
    _NEG,
    _PROFILE_KIND_COLUMNS,
    _VEC_CACHE,
)
# <<< WP1 导入





# 由 dbside.py 提供：_cat, _db_backfill_hash, _db_boost, _db_for, _db_import_json, _db_open, _db_prepare_call_events, _db_ready, _db_rebuild_links, _db_score, _db_sync_call_aggregates, _db_upsert, _db_vec_get, _db_vec_put, _db_where_time, _edge_type, _entry_fact_reference, _entry_row, _fact_references, _is_excluded, _is_fact_source, _is_meta_file, _row_entry, db_load_all








# 由 stat_tracker.py 提供：_db_stats_start, _fact_call_stats, db_record_calls




































def db_build(memdir, outpath=None, extra_dirs=None, exclude=None, state_dir=None):
    """增量建库：只重扫 mtime 变过的文件；返回条目总数。
    extra_dirs 由插件传入并限定在角色目录内；每个附加来源使用 facts/ 前缀避免路径碰撞。

    ★state_dir：审计状态与人工操作状态所在的目录。安全重建会先建到临时目录，
      此时库在 tempDir、而 audit_state.json / fact_ops.json 仍在插件实际目录里。
      不显式指定就会"从空状态重放"，把已软删的卡全部复活、人工编辑全部回退。
    """
    state_index = os.path.join(state_dir, "index.json") if state_dir else outpath
    if state_dir and not os.path.isdir(state_dir):
        # 状态目录给错会让审计/人工操作"静默失效"，必须留痕
        _warn("build state directory is missing", RuntimeError(state_dir))
    c = dbside._db_for(outpath)
    if c is None:
        return None
    _db_import_json(c, outpath)
    _db_backfill_hash(c)
    if not memdir or not os.path.isdir(memdir):
        # 目录不存在（配置错/盘没挂）时不要当成"记忆全没了"去清库
        return c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    skip_dirs = {"settled", "backup", "备份", "archive", "MemoryFacts"}
    ex = set(str(x).replace("\\", "/") for x in (exclude or []))
    jobs = [(memdir, "")]
    used_prefix = {""}
    for d in (extra_dirs or []):
        if d and os.path.isdir(d):
            # ★只有 MemoryFacts 才叫 facts/；其它记忆根用目录名当前缀。
            #   否则副记忆根会被误当成"我们自己的事实层"，压缩逻辑永远跳过它们。
            base_name = os.path.basename(os.path.normpath(d))
            pref = "facts/" if base_name.lower() == "memoryfacts" else base_name + "/"
            if pref in used_prefix:
                i = 2
                while "%s%d/" % (base_name, i) in used_prefix:
                    i += 1
                pref = "%s%d/" % (base_name, i)
            used_prefix.add(pref)
            jobs.append((d, pref))
    seen = set()
    changed = False
    for base, prefix in jobs:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
            for fn in files:
                if not fn.lower().endswith((".md", ".txt")) or ".bak" in fn.lower():
                    continue
                if _is_meta_file(fn):
                    continue        # 自己的审计报告/水位/索引等，不是记忆
                fpath = os.path.join(root, fn)
                inner = os.path.relpath(fpath, base).replace("\\", "/")
                rel = prefix + inner
                if _is_excluded(inner, ex) or _is_excluded(rel, ex):
                    continue
                seen.add(rel)
                try:
                    mt = os.path.getmtime(fpath)
                except OSError as e:
                    _warn('read memory file mtime', e)
                    continue
                row = c.execute("SELECT mtime FROM files WHERE path=?", (rel,)).fetchone()
                if row and abs(float(row["mtime"]) - mt) < 1e-6:
                    continue
                try:
                    text = io.open(fpath, encoding="utf-8", errors="ignore").read()
                except OSError as e:
                    _warn('read memory file', e)
                    continue
                t = _guess_time(fn, fpath)
                c.execute("DELETE FROM entries WHERE source=?", (rel,))
                for i, (head, body) in enumerate(_split_blocks(text)):
                    if len(body.strip()) < 12:
                        continue
                    # ★同一条内容常被写进几十上百天的日记：超过上限只留前几份，
                    #   否则索引会被复制品淹没（实测 813 份同句、重复组 1.3 万）
                    _norm = re.sub(r"\s+", "", body.strip())
                    _th = hashlib.sha1(_norm.encode("utf-8")).hexdigest()[:16]
                    _cap = int(_CFG.get("dup_cap", 3) or 0)
                    if _cap > 0:
                        if c.execute("SELECT COUNT(*) FROM entries WHERE text_hash=?", (_th,)).fetchone()[0] >= _cap:
                            continue
                    hh = _extract_hour(body)
                    ent = {
                        "id": "%s#%d" % (os.path.splitext(rel)[0], i),
                        "anchor": hashlib.sha1(body.strip()[:80].encode("utf-8")).hexdigest()[:12],
                        "source": rel, "time": t["time"], "date": t["date"], "period": t["period"],
                        "hour": hh if hh is not None else t["hour"],
                        "head": head[:60], "summary": _first_sent(body), "text": body.strip(),
                        "_hash": _th,
                    }
                    tags = _tag(body)
                    # 事实卡使用显式的“日期 | 人 | 主题 | 事实”结构。
                    # 将结构化的人名和主题写入索引，避免它们被旧的固定词表漏掉。
                    fact = _parse_fact(body) if prefix == "facts/" else None
                    people = list(tags.get("people") or [])
                    topics = list(tags.get("topics") or [])
                    if fact:
                        if fact.get("who") and fact["who"] not in people:
                            people.append(fact["who"])
                        if fact.get("topic") and fact["topic"] not in topics:
                            topics.append(fact["topic"])
                    ent["people"] = people
                    ent["topics"] = topics
                    ent["things"] = tags.get("things") or []
                    _db_upsert(c, ent)
                c.execute("INSERT OR REPLACE INTO files(path,mtime) VALUES(?,?)", (rel, mt))
                changed = True
    # 清理：凡不在本次扫描集合里的来源都清掉。
    # ★不能只遍历 files 表：从旧 index.json 迁移进来的条目没有 files 行，
    #   被排除/已删除的来源会一直留在库里（排除形同虚设）。
    for r in c.execute("SELECT DISTINCT source FROM entries").fetchall():
        src0 = r["source"] or ""
        if src0 and src0 not in seen:
            c.execute("DELETE FROM entries WHERE source=?", (src0,))
            changed = True
    for r in c.execute("SELECT path FROM files").fetchall():
        p = r["path"]
        if p not in seen:
            c.execute("DELETE FROM files WHERE path=?", (p,))
            changed = True
    if changed:
        _db_rebuild_links(c)
        try:
            c.execute("DELETE FROM vec")
        except sqlite3.Error as e:
            _warn('clear vector table', e)
            pass
        # ★内容变了：所有内存缓存都得失效，否则召回会用旧向量、旧索引、旧图
        _VEC_CACHE.clear()
        _IDX["data"] = None
        _IDX["vocab"] = None
        _IDX["df"] = None
        _GRAPH["built"] = False
        _GRAPH["nodes"] = {}
        _GRAPH["edges"] = {}
    # 审计状态是事实卡索引的持久业务状态，不是一次性页面状态。
    # 每次增量扫描之后都在同一事务里重放它，避免重建把软删卡复活，
    # 或丢掉已经确认的合并结果。状态文件损坏时拒绝继续提交。
    audit_state = _audit_load_state(state_index) if state_index else None
    if audit_state is not None:
        audit_ops._audit_sync_rows(c, state_index, audit_state)
        _db_rebuild_links(c)
    _db_sync_call_aggregates(c)
    c.commit()
    # ★记下当前库对应的索引路径：否则后续 _load_index() 会按旧路径去开库，
    #   把当前连接关掉（直调 db_build 时踩过：Cannot operate on a closed database）
    if outpath:
        _set_index(outpath)
    n = c.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    return n




















































def db_close_op(req):
    """关闭当前数据库连接但不删除文件，供安全换库时释放句柄。"""
    try:
        for state in (_DB, _GRAPH_CACHE):
            cn = state.get("conn")
            if cn is not None:
                try: cn.close()
                except Exception as e:
                    _warn("close database connection", e)
            state.update({"ok": False, "path": "", "conn": None})
        return {"text": "数据库连接已关闭"}
    except Exception as e:
        return {"text": "关闭数据库连接失败：%s" % e}


def db_reset_op(req):
    """清空索引库：关连接 → 删 memory.db（官方文件与事实卡都不动）。"""
    idx = req.get("index") or ""
    dbp = os.path.join(os.path.dirname(idx), "memory.db") if idx else ""
    try:
        for state in (_DB, _GRAPH_CACHE):
            cn = state.get("conn")
            if cn is not None:
                try:
                    cn.close()
                except Exception as e:
                    _warn('close database connection before reset', e)
                    pass
            state.update({"ok": False, "path": "", "conn": None})
        if dbp and os.path.isfile(dbp):
            os.remove(dbp)
        # ★清库也要清缓存，否则 reset 后仍可能从 _IDX/_GRAPH 读到旧数据
        _VEC_CACHE.clear()
        _IDX["data"] = None
        _IDX["vocab"] = None
        _IDX["df"] = None
        _GRAPH["built"] = False
        _GRAPH["nodes"] = {}
        _GRAPH["edges"] = {}
        return {"count": 0, "text": "索引库已清空（下次 build 会重建）"}
    except Exception as e:
        return {"count": 0, "text": "清空失败：%s" % e}


def db_build_op(req):
    """build 的数据库入口：成功后清掉内存缓存，让后续读取走库。"""
    out = req.get("out") or ""
    n = db_build(req.get("memdir") or "", out, req.get("extra") or [], req.get("exclude") or [],
                 req.get("state_dir") or req.get("factsdir") or None)
    if n is None:
        return None
    _set_index(out)
    _IDX["data"] = None
    return {"count": n, "out": out, "text": "建库完成：%d 条" % n}


_DB_OPS = {
    "build": db_build_op,
    "recall": db_recall,
    "record_calls": db_record_calls,
    "bywhen": db_bywhen,
    "debug": db_debug,
    "browse": db_browse,
    "stats": db_stats,
    "links": db_links,
    "reset": db_reset_op,
    "close": db_close_op,
    "graphdata": db_graphdata,
}

def main():

    def reply(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    # 启动先试着把模型装上（装不上也不退出，recall 还能用）
    # --no-model：语义关闭时不加载模型（省启动时间与显存）；真要 embed 时会惰性加载
    if "--no-model" in sys.argv:
        reply({"id": 0, "fn": "ready", "ok": True, "device": "disabled", "index": _find_index()})
    else:
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
        except ValueError as e:
            _warn('parse sidecar request line', e)
            continue
        fn = req.get("fn")
        rid = req.get("id", 0)
        if fn == "__shutdown__":
            break
        try:
            # ★数据库优先：build/recall/browse/stats/links 直接走 SQLite
            _set_index(req.get("index"))
            # 管理页请求必须先走只读连接；不能被 _DB_OPS 的通用 browse
            # 分派截获，否则 _db_ready 可能建表或导入 JSON。
            if fn == "browse" and req.get("readonly"):
                payload = db_browse_request(req)
                if payload is None:
                    reply({"id": rid, "ok": False, "error": "read-only index unavailable"})
                    continue
                reply({"id": rid, "ok": True, **payload})
                continue
            if fn == "profile" and req.get("readonly"):
                payload = db_profile_readonly(req)
                if payload is None:
                    reply({"id": rid, "ok": False, "error": "read-only profile index unavailable"})
                    continue
                if payload.get("invalid"):
                    reply({"id": rid, "ok": False, "error": payload.get("error", "invalid profile request")})
                    continue
                reply({"id": rid, "ok": True, **payload})
                continue
            if fn == "entities" and req.get("readonly"):
                payload = readonly_query.db_entities_readonly(req)
                if payload is None:
                    reply({"id": rid, "ok": False, "error": "read-only entity list unavailable"})
                    continue
                if payload.get("invalid"):
                    reply({"id": rid, "ok": False, "error": payload.get("error", "invalid entity request")})
                    continue
                reply({"id": rid, "ok": True, **payload})
                continue
            _dbop = _DB_OPS.get(fn)
            if _dbop is not None:
                payload = _dbop(req)
                if payload is not None:
                    reply({"id": rid, "ok": True, **payload})
                    continue
            if fn == "ping":
                reply({"id": rid, "ok": True, "device": _MODEL["device"], "dim": 512})
            elif fn == "warm":
                # 预热：后台把模型/CUDA 弄热。★DB 模式不要再把 5 万条载进内存——
                #   召回走 SQL，用不到全量；载进来只会白吃几十上百 MB。
                _set_index(req.get("index"))
                # 图谱伴生服务用只读 warm：只回协议确认，绝不建库/建表/迁移。
                if req.get("readonly"):
                    reply({"id": rid, "ok": True, "count": 0, "readonly": True})
                    continue
                _wc = dbside._db_for(req.get("index"))
                if _wc is None:
                    indexstore._load_index()
                    _vocab()
                    _df()
                # 再吃掉模型/CUDA 的"首次编码"开销：空转一次 embed，真查询就不带这笔了
                if _MODEL.get("ready"):
                    try:
                        _vc = _embed(["预热"])
                        if _vc:
                            _MODEL["warmed"] = True
                    except Exception as e:
                        _warn('warm embedding model', e)
                        pass
                try:
                    _wn = _wc.execute("SELECT COUNT(*) FROM entries").fetchone()[0] if _wc is not None else len(_IDX["data"] or [])
                except sqlite3.Error as e:
                    _warn('count entries for warmup', e)
                    _wn = 0
                reply({"id": rid, "ok": True, "count": _wn})
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
                              skip_recent_days=int(req.get("skip_recent_days") or 0),
                              use_semantic=bool(req.get("use_semantic", False)))
                reply({"id": rid, "ok": True, "bookmarks": to_bookmarks(hits), "count": len(hits),
                       "facts": _fact_references(hits)})
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
                reply({"id": rid, "ok": True, "bookmarks": to_bookmarks(hits), "count": len(hits),
                       "facts": _fact_references(hits)})
            elif fn == "impression":
                _set_index(req.get("index"))
                reply({"id": rid, **impression.impression_get(req)})
            elif fn == "impressionsave":
                _set_index(req.get("index"))
                reply({"id": rid, **impression.impression_save(req)})
            elif fn == "impressionstale":
                _set_index(req.get("index"))
                reply({"id": rid, **impression.impression_stale(req)})
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
                           int(req.get("top_k") or 8), bool(req.get("use_graph", True)),
                           bool(req.get("use_semantic", False)))
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
                                  int(req.get("max_facts") or 400), req.get("index") or "")
                history = audit_history(req)
                reply({"id": rid, "ok": True, "issues": rows, "count": len(rows),
                       "history": history.get("records", []) if isinstance(history, dict) else []})
            elif fn == "auditauto":
                result = audit_auto(req)
                reply({"id": rid, **result})
            elif fn == "auditundo":
                result = audit_undo(req)
                if result.get("error"):
                    reply({"id": rid, "ok": False, **result})
                else:
                    reply({"id": rid, **result})
            elif fn == "factproposal":
                result = fact_propose(req)
                reply({"id": rid, **result})
            elif fn == "proposalstatus":
                result = proposal_status(req)
                reply({"id": rid, **result})
            elif fn == "auditapply":
                result = audit_ops.audit_apply(req)
                if result.get("error"):
                    reply({"id": rid, "ok": False, **result})
                else:
                    reply({"id": rid, **result})
            elif fn == "factapply":
                # 独立事实卡的人工编辑/软删：写入 fact_ops.json 并立即投影。
                result = fact_ops.fact_apply(req)
                if result.get("error"):
                    reply({"id": rid, "ok": False, **result})
                else:
                    reply({"id": rid, **result})
            elif fn == "factundo":
                # 撤回人工编辑/软删：先改操作记录，再把库重新对齐到有效投影。
                result = fact_ops.mark_undone(req)
                if result.get("error"):
                    reply({"id": rid, "ok": False, **result})
                else:
                    index_path = os.path.abspath(str(req.get("index") or ""))
                    audit_ops._audit_sync_db(index_path, audit_ops._audit_load_state(index_path))
                    reply({"id": rid, **result})
            elif fn == "factmerge":
                r = fact_merge(req.get("lines") or [], req.get("existing") or "", float(req.get("threshold") or 0.5))
                reply({"id": rid, "ok": True, **r})
            elif fn == "browse":
                _set_index(req.get("index"))
                if req.get("readonly"):
                    r = readonly_query.db_browse_readonly(req)
                elif req.get("facts_only"):
                    r = readonly_query.db_browse(req)
                else:
                    r = None
                if r is None:
                    r = browse(req.get("query"), req.get("who"), req.get("topic"), req.get("thing"),
                               req.get("date_from") or None, req.get("date_to") or None,
                               int(req.get("offset") or 0), int(req.get("limit") or 50),
                               bool(req.get("facts_only")))
                reply({"id": rid, "ok": True, **r})
            else:
                reply({"id": rid, "ok": False, "error": "unknown fn: %s" % fn})
        except Exception as e:
            reply({"id": rid, "ok": False, "error": "%s: %s" % (type(e).__name__, e)})


if __name__ == "__main__":
    main()
