"""事实卡的独立编辑/软删：可重放的持久操作记录（MemoryFacts/fact_ops.json）。

设计要点（对应修复记录第八节 F04）：
  * 事实卡正文的权威来源仍是 `事实_*.md`；本模块只保存"人工改动"这一层，
    每次建库都按 (created_at, op_id) 顺序重放，因此增量重建、删库重扫、
    进程重启都不会丢编辑或复活软删。
  * 每条操作带 base_version（改动前正文的 text_hash）。重放时与"当前文件投影"
    的版本比对，不一致只记录冲突、不覆盖，避免把别人的改动冲掉。
  * 文件写入与 audit_state.json 同一纪律：临时文件 + fsync + 原子替换，
    SQLite 提交失败时回滚文件。
"""
import os
import io
import re
import json
import uuid
import hashlib
import datetime

import dbside
from state import _GRAPH, _IDX, _set_index, _warn
from textsim import _audit_card_id, _fact_files, _parse_fact, _split_blocks, _tag

OPS_FILE = "fact_ops.json"
_OPS_VERSION = 1
ACTIONS = ("edit", "retire")


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _ops_path(index_path):
    return os.path.join(os.path.dirname(os.path.abspath(index_path)), OPS_FILE)


def body_version(body):
    """正文版本：与 entries.text_hash 完全同一条公式，页面看到的就是这个值。"""
    norm = re.sub(r"\s+", "", str(body or ""))
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def load_ops(index_path):
    """读取操作记录；缺失视为空。文件损坏时抛错，绝不当成"没有改动"。"""
    if not index_path:
        return {"version": _OPS_VERSION, "ops": []}
    path = _ops_path(index_path)
    if not os.path.isfile(path):
        return {"version": _OPS_VERSION, "ops": []}
    try:
        with io.open(path, encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception as exc:
        # 带上路径：用户要能直接找到并修好/移走这个文件，否则只能看到"重建失败"
        raise RuntimeError("fact operations are unreadable (%s); refusing writes: %s"
                           % (path, exc)) from exc
    if not isinstance(state, dict) or state.get("version") != _OPS_VERSION:
        raise RuntimeError("unsupported fact operations state")
    ops = state.get("ops")
    if not isinstance(ops, list):
        raise RuntimeError("invalid fact operations state: ops")
    for op in ops:
        if (not isinstance(op, dict)
                or not isinstance(op.get("op_id"), str)
                or op.get("action") not in ACTIONS
                or not isinstance(op.get("fact_id"), str)
                or not isinstance(op.get("base_version"), str)):
            raise RuntimeError("invalid fact operation record")
        if op["action"] == "edit" and (
                not isinstance(op.get("new_body"), str)
                or not isinstance(op.get("new_hash"), str)):
            raise RuntimeError("invalid fact edit record")
    state.setdefault("ops", [])
    return state


def save_ops(index_path, state):
    if not index_path:
        raise ValueError("index path is required")
    path = _ops_path(index_path)
    tmp = path + ".tmp-" + uuid.uuid4().hex
    try:
        with io.open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(state, fh, ensure_ascii=False, separators=(",", ":"))
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _restore_file(path, previous):
    if previous is None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    tmp = path + ".rollback-" + uuid.uuid4().hex
    try:
        with open(tmp, "wb") as fh:
            fh.write(previous)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _merged_records(index_path):
    """读 audit_state.json 里的合并卡（虚拟卡，正文不在事实文件里）。

    只读、损坏时留痕跳过：这条路径只用于扩展可编辑范围，不该让建库因此失败。
    """
    if not index_path:
        return []
    path = os.path.join(os.path.dirname(os.path.abspath(index_path)), "audit_state.json")
    if not os.path.isfile(path):
        return []
    try:
        with io.open(path, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError) as exc:
        _warn("read merged fact cards: %s" % path, exc)
        return []
    merged = state.get("merged") if isinstance(state, dict) else None
    return [item for item in (merged or []) if isinstance(item, dict)]


def _card_map(index_path):
    """当前有效卡的两种身份：条目 ID / 审计卡 ID → 卡信息。

    覆盖事实文件里的卡，以及 audit_state.json 里的合并卡。写入口能改哪些卡，
    重放就必须能还原哪些卡，两者共用这一份映射才不会出现"改了但重建后丢失"。
    """
    by_entry, by_card = {}, {}
    if not index_path:
        # 没有索引路径就没有事实目录：绝不能按 cwd 的上级目录去猜
        return by_entry, by_card
    factsdir = os.path.dirname(os.path.abspath(index_path))
    if not factsdir or not os.path.isdir(factsdir):
        return by_entry, by_card
    for path in _fact_files(factsdir):
        source = "facts/" + os.path.relpath(path, factsdir).replace("\\", "/")
        try:
            with io.open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeError):
            continue
        ordinal = 0
        for block_index, (_head, body) in enumerate(_split_blocks(text)):
            body = body.strip()
            if not _parse_fact(body):
                continue
            card_id, _version = _audit_card_id(source, body, ordinal)
            ordinal += 1
            entry_id = "%s#%d" % (os.path.splitext(source)[0], block_index)
            info = {"entry_id": entry_id, "card_id": card_id, "source": source, "body": body}
            by_entry[entry_id] = info
            by_card[card_id] = info
    for item in _merged_records(index_path):
        merged_id = str(item.get("id") or "")
        body = str(item.get("body") or item.get("line") or "").strip()
        if not merged_id or not body:
            continue
        info = {"entry_id": merged_id, "card_id": merged_id,
                "source": str(item.get("source") or ""), "body": body}
        by_entry.setdefault(merged_id, info)
        by_card.setdefault(merged_id, info)
    return by_entry, by_card


def _entry_fields(body):
    """按正文重算一条事实条目的可检索字段（与建库时的口径一致）。"""
    parsed = _parse_fact(body) or {"date": "", "who": "", "topic": "", "fact": body}
    tags = _tag(body)
    people = list(tags.get("people") or [])
    topics = list(tags.get("topics") or [])
    things = list(tags.get("things") or [])
    if parsed.get("who") and parsed["who"] not in people:
        people.append(parsed["who"])
    if parsed.get("topic") and parsed["topic"] not in topics:
        topics.append(parsed["topic"])
    head = str(parsed.get("topic") or "")[:60]
    summary = str(parsed.get("fact") or "")[:300]
    hay = " ".join([head, summary, body, " ".join(people + topics + things)]).lower()
    return {
        "head": head, "summary": summary, "hay": hay,
        "people": "|".join(people), "topics": "|".join(topics), "things": "|".join(things),
    }


def _write_row(c, entry_id, body, retired, retired_at):
    fields = _entry_fields(body)
    c.execute(
        "UPDATE entries SET body=?,head=?,summary=?,hay=?,text_hash=?,people=?,topics=?,things=? "
        "WHERE id=?",
        (body, fields["head"], fields["summary"], fields["hay"], body_version(body),
         fields["people"], fields["topics"], fields["things"], entry_id),
    )
    if retired:
        c.execute("UPDATE entries SET retired=1,retired_at=COALESCE(retired_at,?) WHERE id=?",
                  (retired_at, entry_id))
    else:
        # 双向同步：撤回软删后必须真的回到可用状态，否则"撤回"只是删了记录
        c.execute("UPDATE entries SET retired=0,retired_at=NULL WHERE id=?", (entry_id,))


def _ordered_ops(state):
    """按 (created_at, op_id) 排序的有效操作；已撤回的跳过。"""
    ops = [op for op in (state.get("ops") or []) if not op.get("undone_at")]
    return sorted(ops, key=lambda item: (str(item.get("created_at") or ""),
                                         str(item.get("op_id") or "")))


def audit_version(body):
    """审计用的正文版本：与 `_audit_card_id` 的第二返回值同一公式。"""
    return hashlib.sha256(str(body).strip().encode("utf-8")).hexdigest()


def projection(index_path):
    """当前有效投影：文件卡 / 合并卡 + 按顺序重放的人工操作。

    这是**唯一**的真相来源，审计扫描、审计提交、建库重放都必须用它，
    否则会出现"审计还在按旧正文合并""合并卡改了但重建后丢失"这类裂缝。

    返回 (cards, conflicts)。cards 以条目 ID 为键，字段：
      body        当前有效正文（文件正文 + 人工编辑）
      version     正文版本，与 entries.text_hash 同一公式
      audit_version 审计版本，与 _audit_card_id 同一公式
      retired     是否被人工软删
      edited      是否被人工编辑过
      card_id/source/retired_at
    """
    state = load_ops(index_path)          # 损坏时抛错：宁可拒绝，不可静默复活
    by_entry, by_card = _card_map(index_path)
    cards = {}
    for entry_id, card in by_entry.items():
        cards[entry_id] = {
            "entry_id": entry_id,
            "card_id": card.get("card_id") or entry_id,
            "source": card.get("source") or "",
            "body": card["body"],
            "retired": False,
            "edited": False,
            "retired_at": "",
        }
    conflicts = []
    for op in _ordered_ops(state):
        fact_id = str(op.get("fact_id") or "")
        card = by_entry.get(fact_id) or by_card.get(fact_id)
        if card is None:
            conflicts.append({"op_id": op.get("op_id"), "fact_id": fact_id,
                              "reason": "fact card is no longer in the fact files"})
            continue
        row = cards[card["entry_id"]]
        if str(op.get("base_version") or "") != body_version(row["body"]):
            conflicts.append({"op_id": op.get("op_id"), "fact_id": fact_id,
                              "reason": "fact body changed after this operation was recorded"})
            continue
        if op.get("action") == "edit":
            row["body"] = str(op.get("new_body") or "")
            row["edited"] = True
        elif op.get("action") == "retire":
            row["retired"] = True
            row["retired_at"] = str(op.get("created_at") or "")
    for card in cards.values():
        card["version"] = body_version(card["body"])
        card["audit_version"] = audit_version(card["body"])
    return cards, conflicts


def effective_cards(index_path):
    """只需要卡状态时的简写。"""
    return projection(index_path)[0]


def replay_rows(c, index_path, retired_entries=None, cards=None):
    """把「当前有效投影」写回 entries；返回 (改动的条目数, 冲突列表)。

    必须在"文件扫描 + audit_state 合并/软删"之后调用，三个入口共用这一处实现。
    双向同步：编辑生效、撤回后回到文件正文；软删生效、撤回后恢复。
    `retired_entries` 是 audit_state 依据版本确认过的软删条目 ID 集合；
    `cards` 可由调用方传入已算好的投影，避免同一轮里重复扫事实文件。
    """
    if cards is None:
        cards, conflicts = projection(index_path)
    else:
        conflicts = []
    if not cards:
        return 0, conflicts
    retired_entries = set(retired_entries or ())
    applied = 0
    for entry_id, card in cards.items():
        row = c.execute("SELECT body,retired,retired_at FROM entries WHERE id=?",
                        (entry_id,)).fetchone()
        if row is None:
            continue
        wanted_retired = 1 if (card["retired"] or entry_id in retired_entries) else 0
        if (str(row["body"] or "") == card["body"]
                and int(row["retired"] or 0) == wanted_retired):
            continue
        _write_row(c, entry_id, card["body"], wanted_retired,
                   card["retired_at"] or str(row["retired_at"] or ""))
        applied += 1
    return applied, conflicts


def mark_undone(req):
    """撤回一条人工操作：只改操作文件，投影由调用方重新同步。"""
    index_path = os.path.abspath(str(req.get("index") or ""))
    factsdir = os.path.abspath(str(req.get("factsdir") or ""))
    if (not index_path or not factsdir or not os.path.isdir(factsdir)
            or os.path.dirname(index_path) != factsdir
            or os.path.basename(index_path).lower() != "index.json"):
        return {"error": "plugin-owned facts directory unavailable"}
    try:
        op_id = str(uuid.UUID(str(req.get("op_id") or "").strip().lower()))
    except (ValueError, AttributeError):
        return {"error": "op_id must be a UUID"}
    state = load_ops(index_path)
    operation = next((op for op in state["ops"] if str(op.get("op_id") or "") == op_id), None)
    if operation is None:
        return {"error": "fact operation not found", "conflict": True}
    if operation.get("undone_at"):
        return {"ok": True, "idempotent": True, "operation": operation}
    operation["undone_at"] = _now()
    save_ops(index_path, state)
    return {"ok": True, "idempotent": False, "operation": operation}


def fact_apply(req):
    """执行一次人工编辑或软删：写操作记录 → 立即投影到当前库。"""
    index_path = os.path.abspath(str(req.get("index") or ""))
    factsdir = os.path.abspath(str(req.get("factsdir") or ""))
    action = str(req.get("action") or "").strip().lower()
    if action not in ACTIONS:
        return {"error": "invalid fact action"}
    if (not index_path or not factsdir or not os.path.isdir(factsdir)
            or os.path.dirname(index_path) != factsdir
            or os.path.basename(index_path).lower() != "index.json"):
        return {"error": "plugin-owned facts directory unavailable"}
    op_id = str(req.get("op_id") or "").strip().lower()
    try:
        op_id = str(uuid.UUID(op_id))
    except (ValueError, AttributeError):
        return {"error": "op_id must be a UUID"}
    fact_id = str(req.get("fact_id") or "").strip()
    if not fact_id or len(fact_id) > 512:
        return {"error": "fact_id is required"}
    base_version = str(req.get("base_version") or "").strip()
    if len(base_version) > 128:
        return {"error": "invalid base_version"}
    new_body = str(req.get("new_body") or "").strip()
    if action == "edit":
        parsed = _parse_fact(new_body)
        if not parsed or len(new_body) > 2000:
            return {"error": "edited fact must use the date | person | topic | fact format"}
    source = str(req.get("source") or "manual").strip()[:32] or "manual"
    if source not in ("manual", "automatic"):
        return {"error": "invalid fact operation source"}
    if not os.path.isfile(os.path.join(factsdir, "memory.db")):
        # 打开不存在的库会新建一个空库，把"还没建索引"伪装成"卡不存在"。
        return {"error": "plugin database unavailable; rebuild the index first"}
    c = dbside._db_for(index_path)
    if c is None:
        return {"error": "sidecar database unavailable"}
    # 连接换库后同步当前索引路径：后续 op 的 _load_index() 必须读这份库
    _set_index(index_path)
    state = load_ops(index_path)
    snapshot = {"action": action, "fact_id": fact_id, "base_version": base_version,
                "new_body": new_body if action == "edit" else "", "source": source}
    prior = next((op for op in state["ops"] if str(op.get("op_id") or "") == op_id), None)
    if prior is not None:
        same = all(str(prior.get(key) or "") == str(value) for key, value in snapshot.items())
        if not same:
            return {"error": "op_id was already used for a different request", "conflict": True}
        return {"ok": True, "idempotent": True, "operation": prior}

    # ★写入口可改的卡，必须同时是重放能还原的卡。事实文件已改名/删除、或只在库里
    #   残留的行一律拒绝：否则操作"成功"了，下一次重建就会把它丢掉（复核第四轮 P1-2）。
    by_entry, by_card = _card_map(index_path)
    card = by_entry.get(fact_id) or by_card.get(fact_id)
    if card is None:
        return {"error": "fact card is not in the plugin fact files; refresh and rebuild the index",
                "conflict": True}
    row = c.execute("SELECT id,source,body,retired,text_hash FROM entries WHERE id=?",
                    (card["entry_id"],)).fetchone()
    if row is None:
        return {"error": "fact card is not indexed yet", "conflict": True}
    if not dbside._is_fact_source(row["source"]):
        return {"error": "only plugin-owned fact cards can be changed"}
    if int(row["retired"] or 0):
        return {"error": "fact card is already retired", "conflict": True}
    if base_version != str(row["text_hash"] or ""):
        return {"error": "fact card version changed; refresh and try again", "conflict": True}

    now = _now()
    operation = {
        "op_id": op_id, "action": action, "fact_id": str(row["id"]),
        "base_version": base_version, "created_at": now, "source": source,
    }
    if action == "edit":
        operation["new_body"] = new_body
        operation["new_hash"] = body_version(new_body)
    state["ops"].append(operation)
    path = _ops_path(index_path)
    try:
        with open(path, "rb") as fh:
            previous = fh.read()
    except FileNotFoundError:
        previous = None
    saved = False
    # 默认隔离级别下 SELECT 不会隐式开事务，但同一连接可能带着别的 op 未提交的写，
    # 这时再 BEGIN 会报 "cannot start a transaction within a transaction"。
    if not c.in_transaction:
        c.execute("BEGIN IMMEDIATE")
    try:
        save_ops(index_path, state)
        saved = True
        if action == "edit":
            _write_row(c, operation["fact_id"], operation["new_body"], False, "")
        else:
            _write_row(c, operation["fact_id"], str(row["body"] or ""), True, now)
        dbside._db_rebuild_links(c)
        c.commit()
    except Exception:
        try:
            c.rollback()
        finally:
            if saved:
                _restore_file(path, previous)
        raise
    _invalidate_caches(c)
    return {"ok": True, "idempotent": False, "operation": operation}


def _invalidate_caches(c):
    """编辑/软删后：印象按既有规则失效，索引与图谱缓存一并作废。"""
    try:
        import impression
        impression.sweep_stale(c)
    except Exception as exc:
        _warn("sweep impressions after fact operation", exc)
    _IDX.update({"data": None, "vocab": None, "df": None})
    _GRAPH["built"] = False
    _GRAPH["nodes"] = {}
    _GRAPH["edges"] = {}
