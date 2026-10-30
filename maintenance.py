"""历史修整：先预览、再备份、最后走可撤回的持久操作链。

五件事，风险从低到高（各自独立，只有被选中的动作才生效）：

1. 标签规整（tags）——`people/topics/things` 里的顿号切成竖线、去空、去重。**纯格式**。
2. 噪声剔除（noise）——按统一主体分类把代词丢掉，把软件/事件标签分流到对应类别。
   节点与边是派生数据，由重建重算；**不直接删节点**，孤岛按真实边算（度为 0）。
3. 过期退役（expiry）——只处理「疑似过期」候选，退役旧卡走 `audit_apply` 的软删通道。
4. hay 重写（hay）——历史 `hay` 复制了整段正文；重写成 标题+摘要+标签 并 VACUUM，
   省下的空间按库文件实际大小报告（DB-01）。
5. 存量去重（dedupe）——只合并"正文规范化后完全相同"的条目，保留时间/出处/标签，
   并把"被并入哪一条"记进状态，重建时重放，不会重新长回来（DB-02）。

预览用**只读连接**，不做任何迁移；执行前先打包备份（`maint-backup-<时间>.zip`）。
维护自己的软删/物理清/合并记录写在 `maint_state.json`，建库时重放，见 `apply_state`。
"""
import datetime
import io
import json
import os
import re
import sqlite3
import uuid
import zipfile

import audit_ops
import dbside
import fact_ops
import textsim
from state import _GRAPH, _warn

BACKUP_FILES = ("memory.db", "index.json", "audit_state.json", "fact_ops.json",
                "maint_state.json", "people_confirm.json", "profile_overrides.json")
ACTIONS = ("tags", "noise", "expiry", "hay", "dedupe")
STATE_NAME = "maint_state.json"
STATE_VERSION = 1
_MIN_BODY = 12


def _index_of(factsdir, index_path=""):
    factsdir = os.path.abspath(factsdir or "")
    index_path = os.path.abspath(index_path) if index_path else os.path.join(factsdir, "index.json")
    return factsdir, index_path


def _db_file(index_path):
    return os.path.join(os.path.dirname(os.path.abspath(index_path)), "memory.db")


def _ro_conn(index_path):
    """只读连接：预览绝不能触发迁移或建表。"""
    path = _db_file(index_path)
    if not os.path.isfile(path):
        return None
    try:
        c = sqlite3.connect("file:%s?mode=ro" % path.replace("\\", "/"), uri=True, timeout=5)
        c.row_factory = sqlite3.Row
        return c
    except sqlite3.Error as exc:
        _warn("open read-only database: %s" % path, exc)
        return None


def _has_table(c, name):
    try:
        return c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                         (name,)).fetchone() is not None
    except sqlite3.Error:
        return False


# ---------- 维护状态（软删 / 物理清），建库时重放 ----------

def state_path(factsdir):
    return os.path.join(factsdir, STATE_NAME)


def load_state(factsdir):
    """读维护状态。

    ★文件不存在 = 没有决定（空状态）；文件存在但读不了/解析失败 = 报错。
      损坏时按空状态继续会让所有退役条目重新进入召回（审查 S2），所以必须抛出。
    """
    path = state_path(factsdir)
    empty = {"version": STATE_VERSION, "retired": {}, "purged": {}, "merged": {}}
    if not os.path.isfile(path):
        return empty
    try:
        with io.open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise RuntimeError("maintenance state is unreadable: %s (%s)" % (path, exc))
    if not isinstance(data, dict):
        raise RuntimeError("maintenance state is malformed: %s" % path)
    # 字段存在但类型不对 = 结构损坏：不能拿空集合顶替，否则退役决定被静默丢掉（三轮复核 S2 残留）
    for key in ("retired", "purged", "merged"):
        if key in data and not isinstance(data[key], dict):
            raise RuntimeError("maintenance state field '%s' is malformed: %s" % (key, path))
        data.setdefault(key, {})
    version = data.get("version", STATE_VERSION)
    if not isinstance(version, int):
        raise RuntimeError("maintenance state version is malformed: %s" % path)
    data["version"] = version
    return data


def save_state(factsdir, state):
    path = state_path(factsdir)
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(state, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def _union_tags(c, entry_id, tags):
    """把合并时记录的标签并集写回父条目；已存在的标签不动，重复重放幂等。

    去重合并只改派生列。重建时父条目是重新分类出来的，旧库里的历史标签
    （例如后来补上的共同主体）不会自己回来，所以必须由 maint_state 重放（R08）。
    """
    if not isinstance(tags, dict) or not tags:
        return 0
    row = c.execute("SELECT people, topics, things, people_shared, people_pending "
                    "FROM entries WHERE id=?", (entry_id,)).fetchone()
    if row is None:
        return 0
    union = {"people": textsim.split_tag_list(row["people"]),
             "topics": textsim.split_tag_list(row["topics"]),
             "things": textsim.split_tag_list(row["things"]),
             "shared": textsim.split_tag_list(row["people_shared"]),
             "pending": textsim.split_tag_list(row["people_pending"])}
    for key in ("people", "topics", "things", "shared", "pending"):
        for name in tags.get(key) or []:
            name = str(name or "").strip()
            if name and name not in union[key]:
                union[key].append(name)
    # 再走一次统一分类：用户后来确认的人物要从 pending 升回 people，不能两列并存
    clean = textsim.classify_tags(union)
    updates = {}
    for column, key in (("people", "people"), ("topics", "topics"), ("things", "things"),
                        ("people_shared", "shared"), ("people_pending", "pending")):
        current = textsim.split_tag_list(row[column])
        if clean[key] != current:
            updates[column] = "|".join(clean[key])
    if not updates:
        return 0
    c.execute("UPDATE entries SET %s WHERE id=?" % ", ".join("%s=?" % name for name in updates),
              tuple(updates.values()) + (entry_id,))
    # 标签变了，hay 也要跟着变：否则补回来的共同主体/待分类词无法被关键词预筛命中
    merged_row = c.execute("SELECT id, head, summary, people, topics, things, "
                           "people_shared, people_pending FROM entries WHERE id=?",
                           (entry_id,)).fetchone()
    if merged_row is not None:
        c.execute("UPDATE entries SET hay=? WHERE id=?", (_expected_hay(merged_row), entry_id))
    return 1


def apply_state(c, factsdir, state=None):
    """建库末尾重放维护状态：软删不被复活，物理清的条目不重新出现。

    两条路径都核对正文版本：条目正文（text_hash）与记录不一致时，
    说明这是一条改过的新内容，旧决定不再适用（审查 S1）。
    `state` 可由调用方传入（用于"先改内存状态 → 落库 → 再写文件"的一致流程）。

    ★合并决定还必须核对父端（5.1.1 复核 R02）：父条目被删掉或改成别的内容后，
      被并入的子条目已经不再重复，无条件删除会让一个仍然独立的事实从召回里消失。
      父端缺失／改写时保留子端，并把标签并集重放回父端（R08）。
    """
    if c is None or not factsdir:
        return {"retired": 0, "purged": 0, "merged": 0, "stale": 0}
    if state is None:
        state = load_state(factsdir)
    retired = purged = merged = stale = 0
    # 合并（DB-02）：被并入的条目在重建后可能又被扫回来，按 id + 正文版本再删一次。
    # 先做这一步，后面的退役/物理清就查不到已删的行，不会互相踩。
    for entry_id, info in list((state.get("merged") or {}).items()):
        info = info or {}
        row = c.execute("SELECT text_hash FROM entries WHERE id=?", (entry_id,)).fetchone()
        if row is None:
            continue
        version = str(info.get("version") or "")
        if not version or str(row[0] or "") != version:
            # 正文换过了：这不是当初被合并的那条，保留（与 S1 的版本核对同一原则）
            stale += 1
            continue
        parent_id = str(info.get("parent") or "")
        # 旧记录没有 parent_version 时，用子端版本作父端应保持的内容：合并当时两者相同。
        parent_version = str(info.get("parent_version") or version)
        parent = None
        if parent_id:
            parent = c.execute("SELECT text_hash FROM entries WHERE id=?",
                               (parent_id,)).fetchone()
        if parent_id and (parent is None or str(parent[0] or "") != parent_version):
            # 父端已删除或已改写：子端保留原始内容，重新判断而不是直接隐藏
            stale += 1
            continue
        if parent is not None:
            _union_tags(c, parent_id, info.get("tags") or {})
        c.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        # ★让下一轮增量建库重新读这个来源：否则源文件没改就永远不再扫，
        #   父端后来被改写时子端没法重新判断（R02 的增量路径）。
        if info.get("source"):
            c.execute("DELETE FROM files WHERE path=?", (str(info["source"]),))
        try:
            c.execute("DELETE FROM vec WHERE id=?", (entry_id,))
        except sqlite3.Error:
            pass
        merged += 1
    for entry_id, info in list((state.get("retired") or {}).items()):
        info = info or {}
        row = c.execute("SELECT text_hash FROM entries WHERE id=?", (entry_id,)).fetchone()
        if row is None:
            continue
        version = str(info.get("version") or "")
        if version and str(row[0] or "") != version:
            stale += 1
            continue
        c.execute("UPDATE entries SET retired=1, retired_at=COALESCE(retired_at,?) WHERE id=?",
                  (str(info.get("at") or ""), entry_id))
        retired += 1
    for entry_id, info in list((state.get("purged") or {}).items()):
        info = info or {}
        row = c.execute("SELECT text_hash FROM entries WHERE id=?", (entry_id,)).fetchone()
        if row is None:
            continue
        version = str(info.get("version") or "")
        if not version:
            # 旧版没记版本：无法确认是不是同一条内容，宁可保留（可再清一次）
            stale += 1
            continue
        if str(row[0] or "") != version:
            stale += 1
            continue
        c.execute("DELETE FROM entries WHERE id=?", (entry_id,))
        purged += 1
    return {"retired": retired, "purged": purged, "merged": merged, "stale": stale}


def _invalidate_graph():
    _GRAPH["built"] = False
    _GRAPH["nodes"] = {}
    _GRAPH["edges"] = {}


def _resync_derived(c, factsdir, index_path):
    """改完条目后重算派生图，并让内存里的图缓存失效（审查 S4）。"""
    try:
        dbside._db_rebuild_links(c)
        c.commit()
    except Exception as exc:
        _warn("rebuild links after maintenance change", exc)
    _invalidate_graph()


def retire_entries(factsdir, index_path, entry_ids, reason="maintenance", now=None):
    """软删一批条目：先写状态文件，再同步到库。返回真正处理的条数。"""
    factsdir, index_path = _index_of(factsdir, index_path)
    ids = [str(x) for x in (entry_ids or []) if str(x or "")]
    if not ids:
        return 0
    stamp = now or datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = load_state(factsdir)
    c = dbside._db_for(index_path)
    applied = []
    for entry_id in ids:
        version = ""
        if c is not None:
            row = c.execute("SELECT text_hash FROM entries WHERE id=?", (entry_id,)).fetchone()
            if row is not None:
                version = str(row[0] or "")
        state["retired"][entry_id] = {"version": version, "at": stamp, "reason": reason}
        applied.append(entry_id)
    save_state(factsdir, state)
    if c is not None and applied:
        c.executemany("UPDATE entries SET retired=1, retired_at=COALESCE(retired_at,?) WHERE id=?",
                      [(stamp, entry_id) for entry_id in applied])
        c.commit()
        _resync_derived(c, factsdir, index_path)
    return len(applied)


def purge_entries(factsdir, index_path, entry_ids, now=None):
    """物理清理已经退役够久的条目：库里删行，同时记进状态，重建不会再放回来。

    ★记录必须带正文版本：否则同一 ID 换了一段新正文时，重建会把新内容删掉（审查 S1）。
    """
    factsdir, index_path = _index_of(factsdir, index_path)
    ids = [str(x) for x in (entry_ids or []) if str(x or "")]
    if not ids:
        return 0
    stamp = now or datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = load_state(factsdir)
    c = dbside._db_for(index_path)
    for entry_id in ids:
        version = ""
        if c is not None:
            row = c.execute("SELECT text_hash FROM entries WHERE id=?", (entry_id,)).fetchone()
            if row is not None:
                version = str(row[0] or "")
        state["purged"][entry_id] = {"version": version, "at": stamp}
        state["retired"].pop(entry_id, None)
    save_state(factsdir, state)
    if c is not None:
        c.executemany("DELETE FROM entries WHERE id=?", [(entry_id,) for entry_id in ids])
        c.commit()
        _resync_derived(c, factsdir, index_path)
    return len(ids)


def restore(factsdir, index_path="", entry_ids=None):
    """撤回维护软删：只撤回真实存在的维护决定，并按共同有效投影重新同步。

    ★不能对请求里的 ID 无条件 `retired=0`：那个人工/审计软删的卡会立刻重返召回（审查 S3）。
    ★撤完必须重放「审计 + 人工操作 + 剩余维护决定」三者合成的投影，并在**一个事务里**
      完成：先撤掉的只是维护决定，其他来源的退役不能被顺手抹掉（三轮复核 S3 残留）。
    """
    factsdir, index_path = _index_of(factsdir, index_path)
    previous = load_state(factsdir)
    candidates = list(previous.get("retired") or {}) if entry_ids is None else [str(x) for x in entry_ids]
    removed_ids = [entry_id for entry_id in candidates if entry_id in previous["retired"]]
    if not removed_ids:
        # 没有任何维护决定可撤：一个字节都不改
        return 0
    state = {"version": previous.get("version", STATE_VERSION),
             "retired": {k: v for k, v in (previous.get("retired") or {}).items()
                         if k not in removed_ids},
             "purged": dict(previous.get("purged") or {}),
             "merged": dict(previous.get("merged") or {})}
    c = dbside._db_for(index_path)
    if c is None:
        save_state(factsdir, state)
        return len(removed_ids)
    # ★状态文件与库内投影必须同进同退（四轮复核 P2）：
    #   1) 全程只有一个事务，建图用 commit=False，中途不提交；
    #   2) 先落盘状态文件，失败就回滚库（库仍与旧文件一致）；
    #   3) 落盘成功后再提交；提交失败则把文件写回旧内容（库已回滚，两者仍一致）。
    saved = False
    try:
        # 1) 先把这些条目复位；非事实卡的记忆条目只有这一步会处理到
        c.executemany("UPDATE entries SET retired=0, retired_at=NULL WHERE id=?",
                      [(entry_id,) for entry_id in removed_ids])
        # 2) 审计 + 人工操作投影（事实卡的正文/退役以投影为准）
        audit_state = audit_ops._audit_load_state(index_path)
        audit_ops._audit_sync_rows(c, index_path, audit_state, maintenance_state=state)
        # 3) 剩余维护决定最后重放：上一步只认审计与人工，会把其他维护退役的卡复位
        apply_state(c, factsdir, state)
        # 4) 关系图在全部投影之后再重建，并进同一个事务
        dbside._db_rebuild_links(c, commit=False)
        # 5) 状态先落盘；写失败则走 except 回滚，库保持旧状态
        save_state(factsdir, state)
        saved = True
        c.commit()
    except Exception:
        try:
            c.rollback()
        except Exception as exc:
            _warn("rollback maintenance restore", exc)
        if saved:
            # 库已回滚但文件已是新内容：写回旧内容，别让下一次建库按新文件复活/退役
            try:
                save_state(factsdir, previous)
            except Exception as exc:
                _warn("restore maintenance state after failed commit; manual fix needed: %s"
                      % state_path(factsdir), exc)
        raise
    _invalidate_graph()
    return len(removed_ids)


# ---------- 预览 ----------

def _degrees(c):
    """节点在真实边上的度数。度为 0 才是孤岛，度为 1 是叶节点，要分开报。"""
    deg = {name: 0 for (name,) in c.execute("SELECT name FROM nodes").fetchall()}
    for a, b in c.execute("SELECT a, b FROM edges").fetchall():
        if a in deg:
            deg[a] += 1
        if b in deg:
            deg[b] += 1
    return deg


def _noise_rows(c):
    rows = []
    for row in c.execute("SELECT id, people FROM entries").fetchall():
        raw = textsim.split_tag_list(row["people"])
        dropped = []
        for name in raw:
            # 判不准的中文标签是"移入待分类"，不是删除，不能算噪声（R05）
            if textsim.classify_subject(name)["kind"] in ("pronoun", "empty", "object", "event"):
                dropped.append(name)
        if dropped:
            rows.append({"id": str(row["id"]), "dropped": dropped[:6]})
    return rows


def _tag_expr(c, column):
    """旧库可能还没有 5.1 新增列；预览是只读连接，不能靠迁移补列。"""
    try:
        columns = {row[1] for row in c.execute("PRAGMA table_info(entries)")}
    except sqlite3.Error:
        columns = set()
    return column if column in columns else "''"


def _lob_bytes(c, column):
    try:
        row = c.execute("SELECT COALESCE(SUM(LENGTH(CAST(%s AS BLOB))),0) FROM entries" % column).fetchone()
    except sqlite3.Error:
        return 0
    return int(row[0] or 0)


def _expected_hay(row):
    entry = {"head": row["head"], "summary": row["summary"],
             "people": textsim.split_tag_list(row["people"]),
             "topics": textsim.split_tag_list(row["topics"]),
             "things": textsim.split_tag_list(row["things"]),
             "shared": textsim.split_tag_list(row["people_shared"]),
             "pending": textsim.split_tag_list(row["people_pending"])}
    return textsim.hay_of(entry)


def _hay_plan(c):
    """hay 盘点：多少行还是"正文副本"，重写后能省多少字节（逻辑字段口径）。"""
    outdated = redundant = 0
    shared = _tag_expr(c, "people_shared")
    pending = _tag_expr(c, "people_pending")
    for row in c.execute("SELECT id, head, summary, people, topics, things, %s AS people_shared, "
                         "%s AS people_pending, hay FROM entries" % (shared, pending)):
        want = _expected_hay(row)
        old = str(row["hay"] or "")
        if old == want:
            continue
        outdated += 1
        redundant += max(0, len(old.encode("utf-8")) - len(want.encode("utf-8")))
    return {
        "rows": outdated,
        "hay_bytes": _lob_bytes(c, "hay"),
        "body_bytes": _lob_bytes(c, "body"),
        "reclaimable_bytes": redundant,
    }


def _rewrite_hay(c):
    """把 hay 重写成 标题+摘要+标签；正文只保留 body 一份（DB-01）。"""
    changed = 0
    shared = _tag_expr(c, "people_shared")
    pending = _tag_expr(c, "people_pending")
    rows = c.execute("SELECT id, head, summary, people, topics, things, %s AS people_shared, "
                     "%s AS people_pending, hay FROM entries" % (shared, pending)).fetchall()
    updates = []
    for row in rows:
        want = _expected_hay(row)
        if str(row["hay"] or "") != want:
            updates.append((want, row["id"]))
    for hay, entry_id in updates:
        c.execute("UPDATE entries SET hay=? WHERE id=?", (hay, entry_id))
    changed = len(updates)
    return changed


def _db_size(index_path):
    total = 0
    base = _db_file(index_path)
    for suffix in ("", "-wal", "-shm"):
        try:
            total += os.path.getsize(base + suffix)
        except OSError:
            pass
    return total


def _dup_families(c):
    """按"正文规范化后相同"聚类；组内允许空白差异，但仍是同一段内容。"""
    shared = _tag_expr(c, "people_shared")
    pending = _tag_expr(c, "people_pending")
    rows = c.execute(
        "SELECT id, body, date, time, text_hash, source, retired, people, topics, things, "
        "%s AS people_shared, %s AS people_pending FROM entries "
        "WHERE LENGTH(COALESCE(body,'')) >= ?" % (shared, pending), (_MIN_BODY,)).fetchall()
    families = {}
    for row in rows:
        key = re.sub(r"\s+", "", str(row["body"] or ""))
        families.setdefault(key, []).append(row)
    return rows, [group for group in families.values() if len(group) > 1]


def _near_pairs(rows, budget=40000):
    """近似口径只做盘点、不做合并：同人同主题桶内两两比较，预算用完就停。"""
    buckets = {}
    for row in rows:
        buckets.setdefault((str(row["people"] or ""), str(row["topics"] or "")), []).append(row)
    pairs = []
    for group in buckets.values():
        if len(group) < 2 or len(group) > 60:
            continue        # 超大桶多半是"同一主题的每日记录"，不该当近似重复
        tokens = [textsim._tokens(str(row["body"] or "")) for row in group]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                budget -= 1
                if budget <= 0:
                    return pairs
                if textsim._sim(tokens[i], tokens[j]) >= 0.9:
                    pairs.append((group[i], group[j]))
    return pairs


def _dup_report(c, refs=None):
    rows, families = _dup_families(c)
    exact = [g for g in families if len({str(r["body"]) for r in g}) == 1]
    normalized = [g for g in families if len({str(r["body"]) for r in g}) > 1]
    near = _near_pairs(rows)
    sources = {}
    for group in families:
        for row in group[1:]:
            prefix = str(row["source"] or "").split("/")[0] or "(未知)"
            sources[prefix] = sources.get(prefix, 0) + 1
    # 预览就把"哪些组会被跳过"说清楚，别等执行后才发现只合并了一部分
    refs = refs or set()
    skipped = {"retired": 0, "referenced": 0, "different_date": 0}
    mergeable = 0
    for group in families:
        ids = {str(row["id"]) for row in group}
        if any(int(row["retired"] or 0) for row in group):
            skipped["retired"] += 1
        elif ids & refs:
            skipped["referenced"] += 1
        elif len({str(row["date"] or "") for row in group}) > 1:
            skipped["different_date"] += 1
        else:
            mergeable += 1
    return {
        "exact_groups": len(exact),
        "exact_rows": sum(len(g) for g in exact),
        "normalized_groups": len(normalized),
        "normalized_rows": sum(len(g) for g in normalized),
        "near_pairs": len(near),
        "redundant_rows": sum(len(g) - 1 for g in families),
        "mergeable_groups": mergeable,
        "skipped": skipped,
        "groups": len(families),
        "sources": dict(sorted(sources.items(), key=lambda pair: -pair[1])[:8]),
        "sample": [
            {"ids": [str(row["id"]) for row in group[:4]],
             "source": str(group[0]["source"] or ""),
             "date": str(group[0]["date"] or ""),
             "preview": str(group[0]["body"] or "")[:80]}
            for group in families[:5]
        ],
    }


def _dup_references(factsdir, index_path, c):
    """已被审计/人工/维护/调用统计引用过的条目：不参与合并，避免引用悬空。"""
    refs, sources = set(), set()
    try:
        for op in (fact_ops.load_ops(index_path).get("ops") or []):
            refs.add(str(op.get("fact_id") or ""))
    except (RuntimeError, OSError, ValueError) as exc:
        _warn("read fact operations for dedupe", exc)
    try:
        audit_state = audit_ops._audit_load_state(index_path)
        refs.update(str(key) for key in (audit_state.get("retired") or {}))
        for item in audit_state.get("merged") or []:
            if isinstance(item, dict):
                refs.add(str(item.get("id") or ""))
                if item.get("source"):
                    sources.add(str(item["source"]))
        for operation in (audit_state.get("operations") or {}).values():
            if isinstance(operation, dict):
                refs.add(str(operation.get("target") or ""))
    except (RuntimeError, OSError, ValueError) as exc:
        _warn("read audit state for dedupe", exc)
    try:
        state = load_state(factsdir)
        refs.update(str(key) for key in (state.get("retired") or {}))
        refs.update(str(key) for key in (state.get("purged") or {}))
        refs.update(str(key) for key in (state.get("merged") or {}))
    except RuntimeError as exc:
        _warn("read maintenance state for dedupe", exc)
        raise
    try:
        for (fact_id,) in c.execute("SELECT DISTINCT fact_id FROM call_events"):
            refs.add(str(fact_id or ""))
    except sqlite3.Error:
        pass
    try:
        for (payload,) in c.execute("SELECT fact_ids FROM person_impressions"):
            for fact_id in json.loads(payload or "[]"):
                refs.add(str(fact_id))
    except (sqlite3.Error, ValueError, TypeError):
        pass
    refs.discard("")
    return refs, sources


def preview(factsdir, index_path="", actions=None):
    """只读盘点：会改多少、涉及哪些条目、有哪些候选。

    `actions` 给定时按实际选择标注 `applies`：只选 tags 时，人物标签剔除不计入。
    """
    factsdir, index_path = _index_of(factsdir, index_path)
    if not os.path.isdir(factsdir):
        return {"error": "facts directory unavailable"}
    wanted = [name for name in (actions or ACTIONS) if name in ACTIONS]
    c = _ro_conn(index_path)
    if c is None:
        return {"error": "plugin database unavailable (read-only open failed)"}
    try:
        total = int(c.execute("SELECT COUNT(*) FROM entries").fetchone()[0])
        retired = int(c.execute("SELECT COUNT(*) FROM entries WHERE COALESCE(retired,0)=1").fetchone()[0])
        sep_rows = [str(r["id"]) for r in
                    c.execute("SELECT id FROM entries WHERE instr(people, ?)>0", ("\u3001",)).fetchall()]
        noise = _noise_rows(c)
        bad_nodes = [name for (name,) in c.execute("SELECT name FROM nodes").fetchall()
                     if not textsim._looks_like_entity(name)]
        deg = _degrees(c)
        orphans = [name for name, value in deg.items() if value == 0]
        leaves = [name for name, value in deg.items() if value == 1]
        issues = audit_ops.audit_scan(factsdir, limit=200, index_path=index_path)
        kinds = {}
        for row in issues:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        expiry_ids = [row["issue_id"] for row in issues if row["kind"] == "疑似过期"]
        try:
            refs, _sources = _dup_references(factsdir, index_path, c)
            dup = _dup_report(c, refs)
        except (sqlite3.Error, RuntimeError) as exc:
            _warn("scan duplicate entries", exc)
            dup = {"error": str(exc)}
        hay_plan = _hay_plan(c)
        return {
            "ok": True,
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "read_only": True,
            "actions": wanted,
            "entries": {"total": total, "retired": retired},
            "tags": {"rows": len(sep_rows), "sample_ids": sep_rows[:5], "applies": "tags" in wanted},
            "noise": {"rows": len(noise), "tags": sum(len(r["dropped"]) for r in noise),
                      "samples": noise[:8], "applies": "noise" in wanted},
            "nodes": {"total": len(deg), "noise": len(bad_nodes), "noise_samples": bad_nodes[:8],
                      "orphans": len(orphans), "orphan_samples": orphans[:8],
                      "leaves": len(leaves)},
            "duplicates": {"kinds": kinds, "expiry_issue_ids": expiry_ids[:20],
                           "expiry_total": len(expiry_ids), "applies": "expiry" in wanted},
            "hay": dict(hay_plan, applies="hay" in wanted),
            "dedupe": dict(dup, applies="dedupe" in wanted),
        }
    finally:
        try:
            c.close()
        except sqlite3.Error:
            pass


# ---------- 执行 ----------

def backup(index_path, factsdir):
    """把库与状态文件打包成 zip，放在 facts 目录里（zip 不会被当成记忆源扫描）。"""
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    target = os.path.join(factsdir, "maint-backup-%s.zip" % stamp)
    tmp = target + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in BACKUP_FILES:
            full = os.path.join(factsdir, name)
            if os.path.isfile(full):
                zf.write(full, name)
    os.replace(tmp, target)
    return target


def _rewrite_tags(c, drop_noise=False):
    """规整标签列；`drop_noise` 才做主体分类，所以 tags 与 noise 是两个独立动作。"""
    changed = removed = 0
    for row in c.execute("SELECT id, people, topics, things, people_shared, people_pending "
                         "FROM entries").fetchall():
        if drop_noise:
            # ★已存的共同主体／待分类必须一起参与重算：只喂 people 会把上一轮
            #   写进 people_shared / people_pending 的记录覆盖成空（5.1.1 复核 R03）
            clean = textsim.classify_tags({
                "people": row["people"], "topics": row["topics"], "things": row["things"],
                "shared": textsim.split_tag_list(row["people_shared"]),
                "pending": textsim.split_tag_list(row["people_pending"]),
            })
            shared, pending = clean["shared"], clean["pending"]
        else:
            clean = {key: textsim.split_tag_list(row[key]) for key in ("people", "topics", "things")}
            shared = textsim.split_tag_list(row["people_shared"])
            pending = textsim.split_tag_list(row["people_pending"])
        people = "|".join(clean["people"])
        topics = "|".join(clean["topics"])
        things = "|".join(clean["things"])
        row_shared = "|".join(shared)
        row_pending = "|".join(pending)
        if ((people, topics, things, row_shared, row_pending) ==
                (row["people"] or "", row["topics"] or "", row["things"] or "",
                 row["people_shared"] or "", row["people_pending"] or "")):
            continue
        if drop_noise:
            removed += len(textsim.split_tag_list(row["people"])) - len(clean["people"])
        c.execute("UPDATE entries SET people=?, topics=?, things=?, people_shared=?, people_pending=? "
                  "WHERE id=?",
                  (people, topics, things, row_shared, row_pending, row["id"]))
        changed += 1
    return changed, max(0, removed)


def _retire_expiry(c, factsdir, index_path, issue_ids):
    """退役「疑似过期」里的旧卡：走 audit_apply 的软删通道，可在审计历史撤回。"""
    if not issue_ids:
        return 0, []
    issues = {row["issue_id"]: row for row in
              audit_ops.audit_scan(factsdir, limit=1000, index_path=index_path)
              if row.get("kind") == "疑似过期"}
    done = []
    for issue_id in issue_ids:
        issue = issues.get(str(issue_id))
        if not issue or not issue.get("supersede_card_id"):
            continue
        versions = {str(card["id"]): str(card["version"]) for card in issue.get("cards") or []}
        result = audit_ops.audit_apply({
            "index": index_path, "factsdir": factsdir, "action": "soft_delete",
            "operation_id": str(uuid.uuid4()), "issue_id": issue_id,
            "versions": versions, "card_id": issue["supersede_card_id"], "source": "automatic",
        })
        if result.get("ok"):
            done.append(issue_id)
        else:
            _warn("retire expiry candidate during maintenance: %s" % issue_id,
                  RuntimeError(str(result.get("error") or "unknown")))
    return len(done), done


def _merge_duplicates(c, factsdir, index_path, cross_date=False):
    """合并"正文规范化后完全相同"的条目（DB-02）。

    只做完全重复，不做近似合并（近似只盘点，交人工审计）：
    * 同一日期才算"同一次事件的重复"；跨日期的相同正文默认保留，因为可能真是
      不同日子发生的事（DB-03 要求保留这类真实重复），要合并必须显式 cross_date；
    * 任何一条已退役、被审计/人工/维护/调用统计引用，整组跳过——引用不能悬空；
    * 保留一条（最早的日期/时间，其次 id），把其余条目的时间、出处、标签与
      "被并入哪一条"记进 `maint_state.json`；重建时 `apply_state` 重放，不会长回来。
    """
    previous = load_state(factsdir)
    refs, source_refs = _dup_references(factsdir, index_path, c)
    state = {"version": previous.get("version", STATE_VERSION),
             "retired": dict(previous.get("retired") or {}),
             "purged": dict(previous.get("purged") or {}),
             "merged": dict(previous.get("merged") or {})}
    _rows, families = _dup_families(c)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    groups = merged = 0
    skipped = {"retired": 0, "referenced": 0, "different_date": 0}
    saved = False
    try:
        for group in families:
            if any(int(row["retired"] or 0) for row in group):
                skipped["retired"] += 1
                continue
            ids = {str(row["id"]) for row in group}
            if ids & refs or {str(row["source"] or "") for row in group} & source_refs:
                skipped["referenced"] += 1
                continue
            dates = {str(row["date"] or "") for row in group}
            if len(dates) > 1 and not cross_date:
                skipped["different_date"] += 1
                continue
            canonical = sorted(group, key=lambda row: (str(row["date"] or ""),
                                                       str(row["time"] or ""),
                                                       str(row["id"])))[0]
            tags = textsim.classify_tags({
                "people": [name for row in group for name in textsim.split_tag_list(row["people"])],
                "topics": [name for row in group for name in textsim.split_tag_list(row["topics"])],
                "things": [name for row in group for name in textsim.split_tag_list(row["things"])],
                "shared": [name for row in group for name in textsim.split_tag_list(row["people_shared"])],
                # 待分类也必须参与并集：漏传会让"执行当下"就丢掉旧库保存的待分类元数据（S04）
                "pending": [name for row in group for name in textsim.split_tag_list(row["people_pending"])],
            })
            for row in group:
                if str(row["id"]) == str(canonical["id"]):
                    continue
                state["merged"][str(row["id"])] = {
                    "parent": str(canonical["id"]),
                    # 父端绑定：父端正文版本变了/父端被删，重放时不能再删子端（R02）
                    "parent_version": str(canonical["text_hash"] or ""),
                    "version": str(row["text_hash"] or ""),
                    "source": str(row["source"] or ""),
                    "date": str(row["date"] or ""),
                    "time": str(row["time"] or ""),
                    # 标签并集持久化：重建后父端按正文重算，旧库标签不会自己回来（R08）
                    "tags": dict(tags),
                    "at": stamp,
                }
                c.execute("DELETE FROM entries WHERE id=?", (row["id"],))
                # 同 apply_state：来源要在下一轮重扫，父端改写后才能重新判断（R02）
                if row["source"]:
                    c.execute("DELETE FROM files WHERE path=?", (str(row["source"]),))
                try:
                    c.execute("DELETE FROM vec WHERE id=?", (row["id"],))
                except sqlite3.Error:
                    pass
                merged += 1
            _union_tags(c, str(canonical["id"]), tags)
            groups += 1
        if not merged:
            c.rollback()
            return {"groups": 0, "merged": 0, "skipped": skipped}
        save_state(factsdir, state)
        saved = True
        c.commit()
    except Exception:
        try:
            c.rollback()
        except sqlite3.Error as exc:
            _warn("rollback duplicate merge", exc)
        if saved:
            try:
                save_state(factsdir, previous)
            except Exception as exc:
                _warn("restore maintenance state after failed merge; manual fix needed: %s"
                      % state_path(factsdir), exc)
        raise
    _resync_derived(c, factsdir, index_path)
    try:
        # 删行只是把页标成空闲，不 VACUUM 的话文件大小不会回落（DB-02 的"占库"问题仍在）
        c.execute("VACUUM")
    except sqlite3.Error as exc:
        _warn("vacuum after duplicate merge", exc)
    return {"groups": groups, "merged": merged, "skipped": skipped}


def apply(factsdir, index_path="", actions=("tags", "noise"), issue_ids=(), dry_run=False,
          cross_date=False):
    """执行整理。先备份再改；只改派生数据与审计/维护状态，不动事实卡正文。"""
    factsdir, index_path = _index_of(factsdir, index_path)
    wanted = [name for name in (actions or ()) if name in ACTIONS]
    if not wanted:
        return {"error": "no supported action requested"}
    plan = preview(factsdir, index_path, wanted)
    if not plan.get("ok"):
        return plan
    if dry_run:
        return {"ok": True, "dry_run": True, "plan": plan, "actions": wanted}
    try:
        backup_path = backup(index_path, factsdir)
    except OSError as exc:
        return {"error": "backup failed: %s" % exc}
    c = dbside._db_for(index_path)
    if c is None:
        return {"error": "plugin database unavailable"}
    summary = {"ok": True, "backup": backup_path, "actions": wanted,
               "before": {"tags": plan["tags"]["rows"], "noise": plan["noise"]["rows"],
                          "expiry": plan["duplicates"]["expiry_total"],
                          "hay": plan["hay"]["rows"], "dedupe": plan["dedupe"].get("redundant_rows", 0)}}
    if "tags" in wanted or "noise" in wanted:
        changed, removed = _rewrite_tags(c, drop_noise=("noise" in wanted))
        c.commit()
        dbside._db_rebuild_links(c)
        c.commit()
        summary["tags"] = {"entries_changed": changed, "tags_removed": removed}
    if "expiry" in wanted:
        count, done = _retire_expiry(c, factsdir, index_path, issue_ids)
        summary["expiry"] = {"retired": count, "issue_ids": done}
    if "hay" in wanted:
        summary["hay"] = _compact_hay(c, index_path)
    if "dedupe" in wanted:
        result = _merge_duplicates(c, factsdir, index_path, cross_date=cross_date)
        summary["dedupe"] = result
        if result["merged"]:
            summary["tag"] = "duplicates merged; rebuild replays maint_state.merged"
    summary["undo"] = ("标签规整与节点重建是派生数据，重建即恢复；"
                       "过期退役在审计历史里逐条撤回；维护软删用 maintenance.restore 撤回；"
                       "去重与 hay 重写已先打包备份，合并记录在 maint_state.json 的 merged 里。")
    return summary


def _compact_hay(c, index_path):
    """重写 hay 并 VACUUM，报告真实回收的库体积（DB-01）。"""
    before_size = _db_size(index_path)
    changed = _rewrite_hay(c)
    c.commit()
    try:
        c.execute("VACUUM")
    except sqlite3.Error as exc:
        _warn("vacuum after hay rewrite", exc)
    after_size = _db_size(index_path)
    return {"rows": changed, "before_bytes": before_size, "after_bytes": after_size,
            "reclaimed_bytes": max(0, before_size - after_size)}
