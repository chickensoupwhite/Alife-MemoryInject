"""人物印象的存储、缓存与失效。

失效策略：印象记录保存生成时依据的事实卡 id 与正文版本；读取时逐条比对当前库，
任何一条事实卡被软删、合并替代或正文变更，就判定过期。这样不依赖"在某个提交入口
记得挂钩子"——撤回、历史数据、未来的新写入路径都自动覆盖。
审计提交后另做一次清扫，让页面在用户查看前就能看到"已过期"。
"""
import datetime
import io
import json

import dbside
from state import _DB_SCHEMA, _warn


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _facts_for_person(c, name):
    """该人物当前有效的事实卡（排除 retired、限于事实卡来源）。"""
    if not name:
        return []
    pattern = "%|" + str(name).strip() + "|%"
    rows = c.execute(
        "SELECT id, body, summary, source, text_hash, date "
        "FROM entries WHERE COALESCE(retired,0)=0 "
        "AND ('|' || COALESCE(people,'') || '|') LIKE ? "
        "ORDER BY date DESC, time DESC",
        (pattern,),
    ).fetchall()
    facts = []
    for row in rows:
        if not dbside._is_fact_source(row["source"]):
            continue
        facts.append({
            "id": str(row["id"] or ""),
            "version": str(row["text_hash"] or ""),
            "text": str(row["body"] or row["summary"] or ""),
            "source": str(row["source"] or ""),
            "date": str(row["date"] or ""),
        })
    return facts


def _is_outdated(c, name, stored_ids, stored_versions):
    """依据是否已过期：要求"完整有效事实集合"与生成时完全一致。

    只比对旧依据会漏掉新增事实（本该重新生成），因此比集合而不是比子集。
    """
    stored = sorted((str(fid), str(ver)) for fid, ver in zip(stored_ids, stored_versions))
    current = sorted((fact["id"], fact["version"]) for fact in _facts_for_person(c, name))
    return stored != current


def _load_row(c, name):
    return c.execute(
        "SELECT name, impression_text, fact_ids, fact_versions, generated_at, is_stale, model_source "
        "FROM person_impressions WHERE name=?", (name,)
    ).fetchone()


def _parse(row):
    try:
        ids = json.loads(row["fact_ids"] or "[]")
    except ValueError:
        ids = []
    try:
        versions = json.loads(row["fact_versions"] or "[]")
    except ValueError:
        versions = []
    return ids, versions


def impression_get(req):
    """取印象：命中未过期缓存直接返回，否则返回依据事实供 C# 侧生成。"""
    name = str(req.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "name is required"}
    readonly = bool(req.get("readonly"))
    c = dbside._db_ready(req)
    if c is None:
        return {"ok": False, "error": "plugin database unavailable"}

    row = _load_row(c, name)
    if row is not None:
        ids, versions = _parse(row)
        if not int(row["is_stale"] or 0) and not _is_outdated(c, name, ids, versions):
            return {
                "ok": True, "name": name, "cached": True, "needs_generation": False,
                "impression": str(row["impression_text"] or ""),
                "fact_ids": ids, "generated_at": str(row["generated_at"] or ""),
                "model_source": str(row["model_source"] or ""),
                "facts": _facts_for_person(c, name),
            }
        if not readonly:
            c.execute("UPDATE person_impressions SET is_stale=1 WHERE name=?", (name,))
            c.commit()

    facts = _facts_for_person(c, name)
    if not facts:
        return {"ok": True, "name": name, "cached": False, "needs_generation": False,
                "impression": "", "facts": [], "reason": "该人物暂无有效事实卡"}
    return {"ok": True, "name": name, "cached": False, "needs_generation": True,
            "impression": "", "facts": facts}


def impression_save(req):
    """写回 C# 侧生成的印象；空文本拒绝写入，避免把模型失败当成印象缓存起来。"""
    name = str(req.get("name") or "").strip()
    text = str(req.get("impression") or req.get("impression_text") or "").strip()
    if not name:
        return {"ok": False, "error": "name is required"}
    if not text:
        return {"ok": False, "error": "empty impression; refuse to cache"}
    c = dbside._db_ready(req)
    if c is None:
        return {"ok": False, "error": "plugin database unavailable"}

    facts = _facts_for_person(c, name)
    by_id = {fact["id"]: fact for fact in facts}
    requested_ids = [str(x) for x in (req.get("fact_ids") or [])]
    requested_versions = [str(x) for x in (req.get("fact_versions") or [])]
    if not requested_ids or not requested_versions:
        return {"ok": False, "error": "fact_ids and fact_versions are required"}
    if len(requested_ids) != len(requested_versions):
        return {"ok": False, "error": "fact_ids and fact_versions length mismatch"}
    # ★校验"当前完整有效集合"：生成期间增删卡或改正文，旧输出一律不得落缓存
    current_pairs = sorted((fact["id"], fact["version"]) for fact in facts)
    if sorted(zip(requested_ids, requested_versions)) != current_pairs:
        return {"ok": False, "error": "stale impression input; facts changed since generation"}
    used = [by_id[x] for x in requested_ids if x in by_id]
    if not used:
        return {"ok": False, "error": "no active fact for this person"}

    stamp = _now()
    c.execute(
        "INSERT OR REPLACE INTO person_impressions"
        "(name, impression_text, fact_ids, fact_versions, generated_at, is_stale, model_source) "
        "VALUES(?,?,?,?,?,0,?)",
        (name, text, json.dumps([f["id"] for f in used], ensure_ascii=False),
         json.dumps([f["version"] for f in used], ensure_ascii=False), stamp,
         str(req.get("model_source") or "")),
    )
    c.commit()
    return {"ok": True, "name": name, "generated_at": stamp, "fact_ids": [f["id"] for f in used]}


def impression_stale(req):
    """把某个人的印象标记为过期（页面"刷新"按钮用，不生成新内容）。"""
    name = str(req.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "name is required"}
    c = dbside._db_ready(req)
    if c is None:
        return {"ok": False, "error": "plugin database unavailable"}
    changed = c.execute("UPDATE person_impressions SET is_stale=1 WHERE name=?", (name,)).rowcount
    c.commit()
    return {"ok": True, "name": name, "marked": bool(changed)}


def sweep_stale(c):
    """审计提交后清扫：把所有依据已变化／已失效的印象置为过期。

    不依赖调用方指出"哪张卡变了"，因此撤回、历史数据与新写入路径都能覆盖。
    """
    try:
        rows = c.execute(
            "SELECT name, fact_ids, fact_versions FROM person_impressions WHERE is_stale=0"
        ).fetchall()
    except Exception as e:
        _warn("scan impressions for staleness", e)
        return 0
    marked = 0
    for row in rows:
        ids, versions = _parse(row)
        if _is_outdated(c, row["name"], ids, versions):
            c.execute("UPDATE person_impressions SET is_stale=1 WHERE name=?", (row["name"],))
            marked += 1
    if marked:
        c.commit()
    return marked


def ensure_schema(c):
    """建表语句在 state._DB_SCHEMA 中；此处只做存在性检查，供旧库兼容确认。"""
    row = c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='person_impressions'"
    ).fetchone()
    return row is not None
