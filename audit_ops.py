"""审计扫描、合并提交、软删、撤回与提案队列。"""

import os
import io
import re
import json
import datetime
import hashlib
import uuid


# >>> WP1 导入（自动生成，勿手改）
import dbside
import fact_ops
import impression
from dbside import _db_rebuild_links, _db_sync_call_aggregates, _db_upsert
from state import _GRAPH, _IDX, _NEG, _warn
from textsim import _audit_card_id, _fact_files, _parse_fact, _sim, _split_blocks, _tag, _tokens
from textsim import fact_date_key, fact_time, looks_like_state
# <<< WP1 导入

def _audit_load_state(index_path):
    empty = {"version": 1, "retired": {}, "resolved": {}, "merged": [], "operations": {},
             "pending_proposals": [], "accepted_sources": {}, "accepted_provenance": {},
             "auto_suppressed": {}}
    if not index_path:
        return empty
    path = os.path.join(os.path.dirname(os.path.abspath(index_path)), "audit_state.json")
    if not os.path.isfile(path):
        return empty
    try:
        with io.open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("unsupported audit state")
        for key in ("retired", "resolved", "operations"):
            if not isinstance(state.get(key), dict):
                raise ValueError("invalid audit state: " + key)
        if not isinstance(state.get("merged"), list):
            raise ValueError("invalid audit state: merged")
        if not isinstance(state.get("pending_proposals", []), list):
            raise ValueError("invalid audit state: pending_proposals")
        if not isinstance(state.get("accepted_sources", {}), dict):
            raise ValueError("invalid audit state: accepted_sources")
        if not isinstance(state.get("accepted_provenance", {}), dict):
            raise ValueError("invalid audit state: accepted_provenance")
        if not isinstance(state.get("auto_suppressed", {}), dict):
            raise ValueError("invalid audit state: auto_suppressed")
        state.setdefault("pending_proposals", [])
        state.setdefault("accepted_sources", {})
        state.setdefault("accepted_provenance", {})
        state.setdefault("auto_suppressed", {})
        return state
    except Exception as exc:
        raise RuntimeError("audit state is unreadable; refusing writes: %s" % exc) from exc


def _audit_save_state(index_path, state):
    if not index_path:
        raise ValueError("index path is required")
    path = os.path.join(os.path.dirname(os.path.abspath(index_path)), "audit_state.json")
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


def _audit_restore_state(path, previous):
    """Restore the prior JSON state if the SQLite commit for its projection fails."""
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


def _audit_target_card_identity(path, source, line):
    """Return the stable identity of a newly accepted fact-card line."""
    try:
        with io.open(path, encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeError):
        return None, None
    ordinal = 0
    match = None
    for _, body in _split_blocks(text):
        body = body.strip()
        if not _parse_fact(body):
            continue
        if body == str(line).strip():
            match = _audit_card_id(source, body, ordinal)
        ordinal += 1
    return match if match is not None else (None, None)


def _audit_fact_records(memdir, index_path="", max_facts=400, offset=0, coverage=None):
    """读取候选事实卡；默认只看最新的一窗，coverage 回报实际扫描范围（R10）。

    offset 表示"已经看过的最新 N 条"，用于继续向更早的记录扫描：
    窗口为 records[total-offset-max_facts : total-offset]，一次不会被全库拖死。
    """
    state = _audit_load_state(index_path) if index_path else {
        "retired": {}, "merged": [], "resolved": {}, "operations": {},
        "pending_proposals": [], "accepted_sources": {}, "accepted_provenance": {}}
    records = []
    # 人工编辑/软删必须先进入"有效投影"：审计的候选正文、相似度、版本、合并输入
    # 和出处全部取自这里。否则仍会拿事实文件里的旧正文去自动合并，
    # 把人工纠正后的卡当成重复项退休掉（复核第四轮 P1-1）。
    cards, _conflicts = fact_ops.projection(index_path) if index_path else ({}, [])
    for path in _fact_files(memdir):
        try:
            with io.open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (OSError, UnicodeError):
            continue
        rel = os.path.relpath(path, memdir).replace("\\", "/")
        source = "facts/" + rel
        lines = text.splitlines()
        line_numbers = {}
        for number, raw in enumerate(lines, 1):
            line_numbers.setdefault(raw.strip(), []).append(number)
        fact_ordinal = 0
        line_cursor = {}
        for block_index, (_, body) in enumerate(_split_blocks(text)):
            parsed = _parse_fact(body)
            if not parsed:
                continue
            entry_id = "%s#%d" % (os.path.splitext(source)[0], block_index)
            card = cards.get(entry_id)
            if card is not None:
                if card["retired"]:
                    fact_ordinal += 1      # 卡序号仍按文件推进，其余卡 ID 保持稳定
                    continue
                effective = str(card["body"] or "").strip()
                if effective and effective != body.strip():
                    body = effective
                    parsed = _parse_fact(body) or parsed
            body = body.strip()
            card_id, version = _audit_card_id(source, body, fact_ordinal)
            fact_ordinal += 1
            occurrences = line_numbers.get(body, [])
            cursor = line_cursor.get(body, 0)
            line_no = occurrences[min(cursor, len(occurrences) - 1)] if occurrences else block_index + 1
            line_cursor[body] = cursor + 1
            retired = state.get("retired", {}).get(card_id)
            if retired and str(retired.get("version") or "") == version:
                continue
            accepted_ref = state.get("accepted_provenance", {}).get(card_id)
            if isinstance(accepted_ref, dict):
                accepted_version = str(accepted_ref.get("version") or "")
                accepted_refs = accepted_ref.get("provenance")
                if accepted_version == version and isinstance(accepted_refs, list) and accepted_refs:
                    provenance = list(accepted_refs)
                else:
                    provenance = [{"source": source, "line_number": line_no, "line": body}]
            else:
                provenance = [{"source": source, "line_number": line_no, "line": body}]
            records.append({
                **parsed, "id": card_id, "version": version, "source": source,
                "file": rel, "line_number": line_no, "line": body,
                "entry_id": entry_id,
                "provenance": provenance,
                "virtual": False,
            })
    for merged in state.get("merged", []):
        if not isinstance(merged, dict) or not merged.get("id") or merged.get("id") in state.get("retired", {}):
            continue
        merged_id = str(merged["id"])
        line = str(merged.get("line") or "")
        version = str(merged.get("version") or "")
        card = cards.get(merged_id)
        if card is not None:
            if card["retired"]:
                continue
            if card["edited"]:
                # 合并卡的人工编辑同样要进候选：正文与版本都必须是最新的
                effective = str(card["body"] or "").strip()
                if effective:
                    line = effective
                    version = card["audit_version"]
        parsed = _parse_fact(line)
        if not parsed:
            continue
        records.append({
            **parsed, "id": merged_id, "version": version,
            "source": str(merged.get("source") or ""), "file": str(merged.get("source") or ""),
            "line_number": 0, "line": line,
            "provenance": list(merged.get("provenance") or []), "virtual": True,
        })
    if index_path:
        import maintenance
        maintained = maintenance.load_state(os.path.dirname(os.path.abspath(index_path)))
        by_entry = {str(r.get("entry_id") or r["id"]): r for r in records}
        def content_version(record):
            return hashlib.sha1(re.sub(r"\s+", "", str(record.get("line") or "")).encode("utf-8")).hexdigest()[:16]
        def hidden(record):
            eid = str(record.get("entry_id") or record["id"])
            version = content_version(record)
            for key in ("retired", "purged"):
                decision = maintained.get(key, {}).get(eid)
                if decision and ((key == "retired" and not decision.get("version"))
                                 or str(decision.get("version") or "") == version):
                    return True
            decision = maintained.get("merged", {}).get(eid)
            if decision and str(decision.get("version") or "") == version:
                parent = by_entry.get(str(decision.get("parent") or ""))
                if parent and content_version(parent) == str(decision.get("parent_version") or version):
                    return True
            return False
        records = [record for record in records if not hidden(record)]
    total = len(records)
    try:
        offset = max(0, int(offset or 0))
        cap = max(1, total) if max_facts is None else max(1, int(max_facts))
    except (TypeError, ValueError):
        offset, cap = 0, 400
    end = max(0, total - offset)
    start = max(0, end - cap)
    records = records[start:end]
    if coverage is not None:
        coverage.update({
            "total": total,
            "scanned": len(records),
            "offset": offset,
            "max_facts": cap,
            # 更早还有没看过的记录：不能把"本窗没有候选"说成"全库没有问题"
            "truncated": start > 0,
            "remaining": start,
            "next_offset": (offset + len(records)) if start > 0 else None,
        })
    return records, state


def _audit_issue_id(kind, cards):
    refs = sorted((str(card.get("id") or ""), str(card.get("version") or "")) for card in cards)
    raw = json.dumps([kind, refs], ensure_ascii=False, separators=(",", ":"))
    return "ai:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _expiry_candidate(a, b, kind, same_key):
    """近似内容里「旧值可能已过期」的那一对。

    三条同时满足才升级：同人同主题、日期可比较且不同、较新的一侧是现态。
    只改提示类型，不改合并语义——这类候选不自动执行、不提供合并正文，
    人工确认的动作是「退役旧卡」，仍可撤回。事件类事实一律不受影响。

    返回 (kind, older_card)；不是过期对就原样返回 (kind, None)。
    """
    if not same_key or kind not in ("重复", "可能重复"):
        return kind, None
    key_a, key_b = fact_date_key(a.get("date")), fact_date_key(b.get("date"))
    if key_a is None or key_b is None or key_a == key_b:
        return kind, None
    newer, older = (a, b) if key_a > key_b else (b, a)
    if not looks_like_state(newer.get("topic"), newer.get("fact")):
        return kind, None
    return "疑似过期", older


_KIND_RANK = {"疑似过期": 0, "重复": 1}

# 页面看到的是"某一窗"的候选；提交时必须能在同一窗里重新定位，不能只重扫最新 1000 条
# （5.1.2 复核 S02）。这里缓存 issue_id → 窗口，供 audit_apply 复算；进程重启后失效时
# 再按请求里的 offset 或逐个旧窗兜底。
_AUDIT_WINDOWS = {}
_AUDIT_WINDOW_TTL = 3600
_AUDIT_WINDOW_MAX = 4000

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _key_fact_difference(a, b):
    """文本高度相似但关键事实不同：否定极性或数字不一致，返回原因（无差异返回 ""）。

    只改候选类型，不判断谁对谁错：这种对必须留给人工，不能因为相似度高就自动合并
    （5.1.1 复核 R01：长句只多一个"不"也可能被判成 0.93 的重复）。
    """
    fa = str(a.get("fact") or "")
    fb = str(b.get("fact") or "")
    if any(word in fa for word in _NEG) != any(word in fb for word in _NEG):
        return "否定表达不同"
    if sorted(_NUMBER_RE.findall(fa)) != sorted(_NUMBER_RE.findall(fb)):
        return "关键数字不同"
    return ""


def _proposal_relation(a, b, dup=0.85, near=0.5):
    same_key = (a["who"] == b["who"]
                and a["topic"] == b["topic"])
    sim = _sim(a.get('_tokens') or _tokens(a['fact']), b.get('_tokens') or _tokens(b['fact']))
    kind = None
    if sim >= float(dup):
        kind = "重复"
    elif same_key and sim >= float(near):
        kind = "可能重复"
    elif same_key and str(a.get("date") or "") == str(b.get("date") or ""):
        proposal_negative = any(w in a["fact"] for w in _NEG)
        fact_negative = any(w in b["fact"] for w in _NEG)
        if proposal_negative != fact_negative:
            kind = "可能矛盾"
    kind, expired_card = _expiry_candidate(a, b, kind, same_key)
    # 关键事实差异先于"重复"：只有已经被判成"疑似过期"的状态更替可以保留，
    # 其余相似但否定/数字不同的对必须留给人工（R01）
    if kind != "疑似过期" and same_key and sim >= float(near) \
            and _key_fact_difference(a, b):
        kind = "可能矛盾"
    return kind, expired_card


def audit_scan(memdir, limit=30, dup=0.85, near=0.5, max_facts=400, index_path="",
               offset=0, coverage=None, only_issue="", pair_ids=None):
    """Find fact-card candidates with stable IDs and versions; never changes cards."""
    facts, state = _audit_fact_records(memdir, index_path, max_facts, offset, coverage)
    out = []
    # A model proposal is not an active fact card yet, but it can still repeat
    # an existing card.  Surface that pair as an ordinary audit candidate so
    # the human can merge it instead of receiving a dead-end "confirm write"
    # action.  Keep the proposal's input-range provenance on its card; merge
    # will carry it into the resulting card and remove the pending proposal.
    conflicted_proposals = set()
    pending_cards = []
    for proposal in state.get("pending_proposals", []):
        if not isinstance(proposal, dict) or not proposal.get("id"):
            continue
        line = str(proposal.get("line") or "").strip()
        parsed = _parse_fact(line)
        if not parsed:
            continue
        pid = str(proposal["id"])
        pending_cards.append({**parsed, "id": pid, "version": str(proposal.get("version") or ""),
                              "source": str(proposal.get("target") or ""),
                              "file": str(proposal.get("target") or ""), "line_number": 0,
                              "line": line, "entry_id": pid,
                              "provenance": list(proposal.get("provenance") or []),
                              "virtual": True, "proposal": True,
                              "mark_source": str(proposal.get("mark_source") or "")})
    for proposal_card in pending_cards:
        proposal_id = str(proposal_card.get("id") or "")
        if pair_ids is not None and proposal_id not in pair_ids:
            continue
        pending_record = next((p for p in state.get('pending_proposals', []) if p.get('id') == proposal_id), {})
        proposal_facts = facts
        if pending_record.get('auto_review') and max_facts is not None:
            proposal_facts, _ = _audit_fact_records(memdir, index_path, max_facts=None)
        for fact in proposal_facts:
            kind, expired_card = _proposal_relation(proposal_card, fact, dup, near)
            if not kind:
                continue
            same_key = proposal_card['who'] == fact['who'] and proposal_card['topic'] == fact['topic']
            sim = _sim(_tokens(proposal_card['fact']), _tokens(fact['fact']))
            issue_id = _audit_issue_id(kind, (proposal_card, fact))
            if only_issue and issue_id != only_issue and only_issue != "ap:" + proposal_id:
                continue
            if issue_id in state.get("resolved", {}):
                continue
            cards = []
            for card in (fact, proposal_card):
                cards.append({key: card.get(key) for key in (
                    "id", "version", "date", "who", "topic", "fact", "source", "file",
                    "line_number", "line", "entry_id", "provenance", "virtual", "proposal",
                    "mark_source")})
            merge_allowed = same_key and kind not in ("可能矛盾", "疑似过期")
            suggestion = ""
            if merge_allowed:
                date = max(str(fact.get("date") or ""), str(proposal_card.get("date") or ""))
                text = (fact["fact"] if len(fact["fact"]) >= len(proposal_card["fact"])
                        else proposal_card["fact"])
                suggestion = "- %s | %s | %s | %s" % (
                    date, proposal_card["who"], proposal_card["topic"], text)
            out.append({
                "issue_id": issue_id, "kind": kind, "score": round(sim, 3),
                "a": fact["line"], "b": proposal_card["line"],
                "file_a": fact["file"], "file_b": proposal_card["file"],
                "cards": cards, "merge_allowed": merge_allowed,
                "accept_allowed": False, "suggestion": suggestion,
                "supersede_allowed": expired_card is not None,
                "supersede_card_id": str((expired_card or {}).get("id") or ""),
                "proposal": True, "proposal_conflict": True,
                "proposal_source": list(proposal_card.get("provenance") or []),
            })
            conflicted_proposals.add(proposal_id)
    # Remaining proposals (conflicts or explicit review-only input) wait for a human. They
    # use a span provenance (source + offset/excerpt) when an exact line
    # cannot be established; never present that span as a fabricated line.
    for proposal in state.get("pending_proposals", []):
        if not isinstance(proposal, dict) or not proposal.get("id"):
            continue
        line = str(proposal.get("line") or "").strip()
        parsed = _parse_fact(line)
        if not parsed:
            continue
        pid = str(proposal["id"])
        if pair_ids is not None and pid not in pair_ids:
            continue
        card = {**parsed, "id": pid, "version": str(proposal.get("version") or ""),
                "source": str(proposal.get("target") or ""),
                "file": str(proposal.get("target") or ""), "line_number": 0,
                "line": line, "entry_id": pid,
                "provenance": list(proposal.get("provenance") or []), "virtual": True,
                "proposal": True, "mark_source": str(proposal.get("mark_source") or "")}
        if pid in conflicted_proposals:
            continue
        issue_id = "ap:" + pid
        if only_issue and issue_id != only_issue:
            continue
        if issue_id in state.get("resolved", {}):
            continue
        out.append({"issue_id": issue_id, "kind": "新事实待确认", "score": 0.0,
                    "a": line, "b": "", "file_a": card["source"], "file_b": "",
                    "cards": [card], "merge_allowed": False, "accept_allowed": True,
                    "suggestion": line, "proposal": True,
                    "proposal_source": list(proposal.get("provenance") or []),
                    "review_peers": [peer for peer in pending_cards if peer['id'] != pid
                                     and _proposal_relation(card, peer)[0]],
                    "review_reason": str(proposal.get('review_reason') or '')})
    # 提交已有双卡候选时，只比较请求指向的两张有效卡。正文、版本和候选类型
    # 仍从当前投影重新读取和计算；提案与已有事实的冲突检查保持完整。
    if pair_ids is not None:
        facts = [fact for fact in facts if str(fact.get("id") or "") in pair_ids]
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
            elif same_key and str(a.get("date") or "") == str(b.get("date") or ""):
                na = any(w in a["fact"] for w in _NEG)
                nb = any(w in b["fact"] for w in _NEG)
                if na != nb:
                    kind = "可能矛盾"
            kind, expired_card = _expiry_candidate(a, b, kind, same_key)
            # 关键事实差异先于"重复"：长句多一个否定、数字不同，都必须留给人工（R01）。
            # 已被判成"疑似过期"的状态更替保持原类型（同样不自动合并、只提示退役旧卡）。
            if kind != "疑似过期" and same_key and sim >= float(near) \
                    and _key_fact_difference(a, b):
                kind = "可能矛盾"
            if not kind:
                continue
            issue_id = _audit_issue_id(kind, (a, b))
            if only_issue and issue_id != only_issue:
                continue
            if issue_id in state.get("resolved", {}):
                continue
            cards = []
            for fact in (a, b):
                cards.append({key: fact.get(key) for key in (
                    "id", "version", "date", "who", "topic", "fact", "source", "file",
                    "line_number", "line", "entry_id", "provenance", "virtual")})
            suggestion = ""
            merge_allowed = same_key and kind not in ("可能矛盾", "疑似过期")
            if merge_allowed:
                date = max(str(a.get("date") or ""), str(b.get("date") or ""))
                text = a["fact"] if len(a["fact"]) >= len(b["fact"]) else b["fact"]
                suggestion = "- %s | %s | %s | %s" % (date, a["who"], a["topic"], text)
            out.append({
                "issue_id": issue_id, "kind": kind, "score": round(sim, 3),
                "a": a["line"], "b": b["line"], "file_a": a["file"], "file_b": b["file"],
                "cards": cards, "merge_allowed": merge_allowed, "suggestion": suggestion,
                "supersede_allowed": expired_card is not None,
                "supersede_card_id": str((expired_card or {}).get("id") or ""),
            })
            if len(out) >= cap:
                break
    out.sort(key=lambda x: (0 if x.get("proposal") else 1,
                            _KIND_RANK.get(x["kind"], 2), -x["score"]))
    if only_issue:
        return out[:1]
    return out[:max(1, int(limit))]


def _remember_windows(index_path, issues, offset, max_facts):
    if not index_path:
        return
    try:
        index_path = os.path.abspath(index_path)
        stamp = datetime.datetime.now(datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return
    for issue in issues or []:
        issue_id = str(issue.get("issue_id") or "")
        if not issue_id:
            continue
        _AUDIT_WINDOWS[issue_id] = {"index": index_path, "offset": max(0, int(offset)),
                                    "max_facts": max(1, int(max_facts)), "at": stamp}
    if len(_AUDIT_WINDOWS) > _AUDIT_WINDOW_MAX:
        for key in sorted(_AUDIT_WINDOWS, key=lambda k: _AUDIT_WINDOWS[k].get("at", 0))[
                :len(_AUDIT_WINDOWS) - _AUDIT_WINDOW_MAX]:
            _AUDIT_WINDOWS.pop(key, None)


def _cached_window(index_path, issue_id):
    item = _AUDIT_WINDOWS.get(issue_id)
    if not isinstance(item, dict):
        return None
    if str(item.get("index") or "") != os.path.abspath(index_path):
        return None
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    if now - float(item.get("at") or 0) > _AUDIT_WINDOW_TTL:
        _AUDIT_WINDOWS.pop(issue_id, None)
        return None
    return int(item.get("offset") or 0), int(item.get("max_facts") or 400)


def _locate_issue(factsdir, index_path, issue_id, req):
    """按缓存窗口、请求窗口、默认窗口、旧窗兜底的顺序重新定位候选。

    每次都重新读取事实卡并重算 issue_id；版本仍由调用方逐张卡核对，防篡改语义不变。
    """
    windows = []
    cached = _cached_window(index_path, issue_id)
    if cached:
        windows.append(cached)
    try:
        if req.get("offset") is not None:
            windows.append((max(0, int(req.get("offset") or 0)),
                            max(1, int(req.get("max_facts") or 400))))
    except (TypeError, ValueError):
        pass
    windows.append((0, 1000))
    seen = set()
    for offset, max_facts in windows:
        if (offset, max_facts) in seen:
            continue
        seen.add((offset, max_facts))
        rows = audit_scan(factsdir, dup=0.85, near=0.5, max_facts=max_facts,
                          index_path=index_path, offset=offset, only_issue=issue_id)
        if rows:
            return rows[0]
    # 进程重启/cache 过期：继续向更早的窗口找，最多再看 4 窗（不扩成全库穷举）
    offset = 1000
    for _ in range(4):
        coverage = {}
        rows = audit_scan(factsdir, dup=0.85, near=0.5, max_facts=1000,
                          index_path=index_path, offset=offset,
                          coverage=coverage, only_issue=issue_id)
        if rows:
            return rows[0]
        if not coverage.get("truncated"):
            break
        offset += 1000
    return None


def audit_scan_page(memdir, limit=30, dup=0.85, near=0.5, max_facts=400, index_path="",
                    offset=0):
    """页面用审计扫描：候选 + 覆盖范围（scanned/total/remaining/next_offset）。

    接口只给候选、不给覆盖信息时，用户无法区分"没有重复"和"根本没扫到"（R10）。
    """
    coverage = {}
    issues = audit_scan(memdir, limit=limit, dup=dup, near=near, max_facts=max_facts,
                        index_path=index_path, offset=offset, coverage=coverage)
    _remember_windows(index_path, issues, coverage.get("offset", offset),
                      coverage.get("max_facts", max_facts))
    return {"issues": issues, "coverage": coverage}


def _audit_plugin_paths(req):
    index_path = os.path.abspath(str(req.get("index") or ""))
    factsdir = os.path.abspath(str(req.get("factsdir") or ""))
    if (not index_path or not factsdir or not os.path.isdir(factsdir)
            or os.path.dirname(index_path) != factsdir
            or os.path.basename(index_path).lower() != "index.json"):
        raise ValueError("plugin-owned facts directory unavailable")
    return index_path, factsdir


def fact_propose(req):
    """Stage model output; auto_accept routes new/duplicate/review cases automatically."""
    try:
        index_path, factsdir = _audit_plugin_paths(req)
    except ValueError as exc:
        return {"error": str(exc)}
    lines = req.get("lines") or []
    if not isinstance(lines, list) or len(lines) > 100:
        return {"error": "proposal lines are invalid"}
    target = os.path.basename(str(req.get("target") or ("事实_" + datetime.date.today().strftime("%Y%m%d") + ".md")))
    if not (target.startswith("事实_") and target.endswith(".md")):
        return {"error": "proposal target is invalid"}
    refs = req.get("provenance") if isinstance(req.get("provenance"), list) else []
    clean = []
    for raw in lines:
        line = str(raw or "").strip()
        if _parse_fact(line):
            clean.append(line)
    if not clean:
        return {"error": "proposal contains no fact lines"}
    state = _audit_load_state(index_path)
    mark_source = str(req.get("mark_source") or "")
    has_pending_source = any(
        isinstance(item, dict) and str(item.get("mark_source") or "") == mark_source
        for item in state.get("pending_proposals", []))
    if mark_source and mark_source in state.get("accepted_sources", {}) and not has_pending_source:
        return {"ok": True, "accepted": True, "proposal_ids": [], "count": 0,
                "deduplicated": True}
    existing = {str(x.get("id")): x for x in state.get("pending_proposals", []) if isinstance(x, dict)}
    existing_by_line = {}
    for item in existing.values():
        if str(item.get("mark_source") or "") != mark_source:
            continue
        item_hash = str(item.get("line_hash") or item.get("version") or "")
        if item_hash:
            existing_by_line[item_hash] = item
    ids = []
    for ordinal, line in enumerate(clean):
        line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
        prior = existing_by_line.get(line_hash)
        if prior is not None:
            ids.append(str(prior.get("id")))
            continue
        raw_id = "%s\0%s\0%s" % (mark_source or target, line_hash, "auto" if req.get("auto_accept") is True else ordinal)
        pid = "ap:" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:32]
        # Keep an independent, immutable copy of the input-range metadata on
        # every proposal.  The range identifies what was supplied to the
        # model; it is not presented as a verified one-to-one source line.
        item_refs = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            copy = dict(ref)
            copy.setdefault("scope", "input-range")
            copy.setdefault("fact_hash", line_hash)
            copy.setdefault("fact_index", ordinal)
            excerpt = str(copy.get("excerpt") or copy.get("text") or "")
            if excerpt and not copy.get("excerpt_hash"):
                copy["excerpt_hash"] = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
            item_refs.append(copy)
        if not item_refs:
            item_refs = [{"scope": "input-range", "fact_hash": line_hash,
                          "fact_index": ordinal}]
        item = {"id": pid, "version": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                "line": line, "line_hash": line_hash, "target": target,
                "provenance": item_refs,
                "mark_source": mark_source,
                "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        if pid not in existing:
            state["pending_proposals"].append(item)
            existing[pid] = item
        existing_by_line[line_hash] = item
        ids.append(pid)
    _audit_save_state(index_path, state)
    if req.get('auto_accept') is True:
        return _auto_ingest_proposals(index_path, factsdir, ids)
    return {"ok": True, "proposal_ids": ids, "count": len(ids), "deduplicated": len(ids) != len(clean)}


def _auto_ingest_proposals(index_path, factsdir, proposal_ids):
    """Auto-write new claims, skip exact duplicates, retain explainable review cases."""
    c = dbside._db_for(index_path)
    if c is None:
        return {'error': 'sidecar database unavailable'}
    active, state = _audit_fact_records(factsdir, index_path, max_facts=None)
    retired = [_parse_fact(row[0]) for row in c.execute(
        "SELECT body FROM entries WHERE retired=1 AND lower(replace(source,char(92),'/')) LIKE 'facts/%'")]
    for card in active:
        card['_tokens'] = _tokens(card['fact'])

    def proposal_card(proposal):
        parsed = _parse_fact(proposal['line'])
        return {**parsed, 'id': proposal['id'], 'version': proposal['version'],
                'line': proposal['line'], 'source': proposal['target'], 'file': proposal['target'],
                'provenance': proposal.get('provenance') or [], 'proposal': True,
                'mark_source': proposal.get('mark_source') or '', 'virtual': True}

    def exact_key(card):
        return tuple(str(card.get(key) or '').strip() for key in ('date', 'who', 'topic')) + (
            re.sub(r'\s+', '', str(card.get('fact') or '')),)

    initial = {p['id']: p for p in state.get('pending_proposals', [])}
    peers = [proposal_card(p) for p in initial.values()]
    for card in peers:
        card['_tokens'] = _tokens(card['fact'])
    unique_ids = list(dict.fromkeys(proposal_ids))
    written, skipped, already = 0, len(proposal_ids) - len(unique_ids), 0
    for pid in unique_ids:
        state = _audit_load_state(index_path)
        proposal = next((p for p in state['pending_proposals'] if p.get('id') == pid), None)
        if proposal is None:
            return {'error': 'proposal changed during automatic ingestion'}
        card = proposal_card(proposal)
        card['_tokens'] = _tokens(card['fact'])
        previous = next((op for op in state.get('operations', {}).values()
                         if not op.get('undone_at') and any(
                             item.get('id') == pid and item.get('version') == card['version']
                             for item in op.get('inputs') or [])), None)
        if previous is not None:
            state['pending_proposals'] = [p for p in state['pending_proposals'] if p.get('id') != pid]
            _audit_save_state(index_path, state)
            already += 1
            continue
        duplicate = next((fact for fact in active if exact_key(fact) == exact_key(card)), None)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        clean_card = {key: value for key, value in card.items() if key != '_tokens'}
        issue_id = 'ap:' + pid
        operation_id = str(uuid.uuid5(uuid.NAMESPACE_URL, 'retromem:auto:' + pid + ':' + card['version']))
        if duplicate is not None:
            _audit_consume_proposals(state, [clean_card], operation_id, now, 'skip_duplicate')
            state['resolved'][issue_id] = {'action': 'skip_duplicate', 'operation_id': operation_id, 'created_at': now}
            state['operations'][operation_id] = {
                'operation_id': operation_id, 'action': 'skip_duplicate', 'issue_id': issue_id,
                'inputs': [clean_card], 'created_at': now, 'source': 'automatic',
                'result': {'id': duplicate['id']}, 'reason': 'same date, subject, topic and normalized fact'}
            _audit_save_state(index_path, state)
            skipped += 1
            continue
        if any(fact and exact_key(fact) == exact_key(card) for fact in retired):
            proposal['auto_review'] = True
            proposal['review_reason'] = '与已退役事实完全相同，不自动恢复，请人工核对'
            _audit_save_state(index_path, state)
            continue
        active_conflict = any(_proposal_relation(card, fact)[0] for fact in active)
        pending_conflict = any(peer['id'] != pid and exact_key(peer) != exact_key(card)
                               and _proposal_relation(card, peer)[0] for peer in peers)
        if active_conflict or pending_conflict:
            proposal['auto_review'] = True
            proposal['review_reason'] = '与有效事实近似或冲突' if active_conflict else '与同批或其他待审提案近似或冲突'
            _audit_save_state(index_path, state)
            continue
        issue = {'issue_id': issue_id, 'proposal': True, 'cards': [clean_card]}
        snapshot = {'action': 'accept', 'issue_id': issue_id, 'card_id': '',
                    'versions': {pid: card['version']}, 'merged': '', 'source': 'automatic'}
        result = _accept_proposal(c, index_path, factsdir, state, issue, snapshot,
                                  operation_id, now, rebuild_links=False, existing_facts=active)
        if not result.get('ok'):
            return result
        identity = result['operation']['result']
        active.append({**card, 'id': identity['id'], 'version': identity['version'],
                       'source': 'facts/' + proposal['target'], 'proposal': False})
        written += 1
    if written or skipped or already:
        _db_rebuild_links(c)
        _db_sync_call_aggregates(c)
        c.commit()
    state = _audit_load_state(index_path)
    pending = [pid for pid in unique_ids if any(p.get('id') == pid for p in state['pending_proposals'])]
    source = str((initial.get(unique_ids[0]) or {}).get('mark_source') or '') if unique_ids else ''
    return {'ok': True, 'proposal_ids': pending, 'count': len(pending),
            'auto_written': written, 'auto_skipped': skipped, 'already_processed': already,
            'pending_count': len(pending), 'processed_count': len(proposal_ids),
            'source_accepted': bool(source and source in state.get('accepted_sources', {}) and not any(
                p.get('mark_source') == source for p in state['pending_proposals']))}


def proposal_status(req):
    """Report pending/accepted state for a plugin-owned source marker."""
    try:
        index_path, _ = _audit_plugin_paths(req)
    except ValueError as exc:
        return {"error": str(exc)}
    state = _audit_load_state(index_path)
    sources = req.get("sources")
    if isinstance(sources, list):
        values = [str(x or "") for x in sources if str(x or "") and len(str(x or "")) <= 512]
        accepted_sources = state.get("accepted_sources", {})
        pending_values = {str(p.get("mark_source") or "") for p in state.get("pending_proposals", [])
                          if isinstance(p, dict) and str(p.get("mark_source") or "")}
        accepted = [value for value in values
                    if value in accepted_sources and value not in pending_values]
        pending = [value for value in values if value in pending_values]
        restored = sorted(value for value in pending_values if value not in set(values))
        return {"ok": True, "accepted_sources": accepted, "pending_sources": pending,
                "restored_sources": restored}
    mark_source = str(req.get("mark_source") or "")
    if not mark_source or len(mark_source) > 512:
        return {"error": "mark_source is required"}
    pending = [p for p in state.get("pending_proposals", [])
               if isinstance(p, dict) and str(p.get("mark_source") or "") == mark_source]
    accepted = state.get("accepted_sources", {}).get(mark_source)
    return {"ok": True, "mark_source": mark_source, "pending": bool(pending),
            "pending_count": len(pending), "accepted": bool(accepted) and not pending,
            "accepted_source": accepted or None}


def _audit_consume_proposals(state, cards, operation_id, now, action):
    """Remove resolved proposal inputs without writing them, retaining source completion."""
    proposal_cards = [card for card in cards if card.get("proposal")]
    if not proposal_cards:
        return
    proposal_ids = {str(card.get("id") or "") for card in proposal_cards}
    state["pending_proposals"] = [
        proposal for proposal in state.get("pending_proposals", [])
        if not (isinstance(proposal, dict)
                and str(proposal.get("id") or "") in proposal_ids)]
    for proposal_card in proposal_cards:
        mark_source = str(proposal_card.get("mark_source") or "")
        if not mark_source:
            continue
        remaining_source = any(
            isinstance(item, dict)
            and str(item.get("mark_source") or "") == mark_source
            for item in state.get("pending_proposals", []))
        # A merge retains the proposal's claim in a reviewed fact card. A
        # keep/soft-delete action discards it, so that source must not be
        # reported as accepted or allowed to advance compaction markers.
        if not remaining_source and action in ("merge", "skip_duplicate"):
            state.setdefault("accepted_sources", {})[mark_source] = {
                "operation_id": operation_id,
                "proposal_id": str(proposal_card.get("id") or ""),
                "action": action,
                "accepted_at": now,
            }


def _accept_proposal(c, index_path, factsdir, state, issue, request_snapshot, operation_id, now,
                     rebuild_links=True, existing_facts=None, review_candidates=None):
    """Shared per-fact file/state/SQLite transaction, with a full active-fact check."""
    action = 'accept'
    issue_id = str(issue['issue_id'])
    cards = issue.get('cards') or []
    if not issue.get("proposal"):
        return {"error": "only pending fact proposals can be accepted"}
    if len(cards) != 1:
        return {"error": "proposal conflicts with an existing fact; review or merge it first",
                "conflict": True, "requires_action": "merge",
                "audit_issue": issue, "candidates": cards}
    proposal_id = str(cards[0].get("id") or "")
    proposal = next((p for p in state.get("pending_proposals", [])
                     if isinstance(p, dict) and str(p.get("id")) == proposal_id), None)
    if proposal is None:
        return {"error": "proposal is stale or no longer available", "conflict": True}
    # A proposal is still subject to the same duplicate/near-duplicate
    # guard as ordinary audit input.  Re-scan immediately before writing
    # so a card added after the proposal was generated cannot become a
    # second active copy.  Same person+topic with materially different
    # facts remains acceptable; only the explainable audit thresholds
    # block direct acceptance and route the user to merge/review.
    parsed_proposal = _parse_fact(str(proposal.get("line") or "").strip())
    if not parsed_proposal:
        return {"error": "proposal fact is invalid", "conflict": True}
    if existing_facts is None:
        existing_facts, _ = _audit_fact_records(factsdir, index_path, max_facts=None)
    parsed_proposal['_tokens'] = _tokens(parsed_proposal['fact'])
    conflicts = []
    for existing in existing_facts:
        if str(existing.get("id") or "") == proposal_id:
            continue
        kind, _ = _proposal_relation(parsed_proposal, existing)
        if kind:
            sim = _sim(_tokens(str(existing.get('fact') or '')), parsed_proposal['_tokens'])
            conflicts.append({"id": existing.get("id"), "version": existing.get("version"),
                              "line": existing.get("line"), "score": round(sim, 3),
                              "kind": kind})
    if conflicts:
        # Return the full audit candidate as well as the compact conflict
        # list.  The HTTP layer can expose this for API clients, while the
        # management page will refresh and render the same merge action.
        conflict_issue = next((row for row in review_candidates or [] if row.get('proposal_conflict')
                               and any(str(card.get('id') or '') == proposal_id for card in row.get('cards') or [])), None)
        return {"error": "proposal conflicts with an existing fact; review or merge it first",
                "conflict": True, "requires_action": "merge", "candidates": conflicts,
                "audit_issue": conflict_issue}
    target = os.path.basename(str(proposal.get("target") or ""))
    if not (target.startswith("事实_") and target.endswith(".md")):
        return {"error": "proposal target is invalid"}
    target_path = os.path.join(factsdir, target)
    before_file = None
    try:
        if os.path.exists(target_path):
            with open(target_path, "rb") as fh:
                before_file = fh.read()
        line = str(proposal.get("line") or "").strip()
        previous = before_file.decode("utf-8") if before_file is not None else ""
        if line not in [x.strip().lstrip("-").strip() for x in previous.splitlines()]:
            if not previous.strip():
                previous = "# 事实卡 · %s\n\n" % datetime.date.today().isoformat()
            if not previous.endswith("\n"):
                previous += "\n"
            previous += (line if line.startswith("-") else "- " + line) + "\n"
            tmp = target_path + ".proposal-" + uuid.uuid4().hex
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(previous)
                fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, target_path)
        state_path = os.path.join(factsdir, "audit_state.json")
        try:
            with open(state_path, "rb") as fh:
                previous_state = fh.read()
        except FileNotFoundError:
            previous_state = None
        state_saved = False
        try:
            c.execute("BEGIN IMMEDIATE")
            # Add a direct projection for immediate management/graph refresh.
            parsed = _parse_fact(line)
            source = "facts/" + target
            accepted_card_id, accepted_version = _audit_target_card_identity(target_path, source, line)
            if not accepted_card_id:
                raise RuntimeError("accepted fact card identity could not be determined")
            tags = _tag(line)
            for name, key in ((parsed["who"], "people"), (parsed["topic"], "topics")):
                if name and name not in tags[key]:
                    tags[key].append(name)
            entry_day, entry_clock = fact_time(parsed["date"])
            entry = {"id": proposal_id, "anchor": hashlib.sha1(line[:80].encode("utf-8")).hexdigest()[:12],
                     "source": source, "time": entry_clock or str(parsed["date"]) + "T12:00:00",
                     "date": entry_day or parsed["date"],
                     "period": "日", "hour": 12, "head": parsed["topic"][:60], "summary": parsed["fact"][:300],
                     "text": line, "people": tags["people"], "topics": tags["topics"], "things": tags["things"],
                     "retired": False, "_hash": hashlib.sha1(re.sub(r"\s+", "", line).encode("utf-8")).hexdigest()[:16]}
            if not _db_upsert(c, entry):
                raise RuntimeError("failed to index accepted fact proposal")
            state["pending_proposals"] = [p for p in state.get("pending_proposals", [])
                                            if not (isinstance(p, dict) and str(p.get("id")) == proposal_id)]
            mark_source = str(proposal.get("mark_source") or "")
            remaining_source = any(
                isinstance(item, dict) and str(item.get("mark_source") or "") == mark_source
                for item in state.get("pending_proposals", []))
            if mark_source and not remaining_source:
                state.setdefault("accepted_sources", {})[mark_source] = {
                    "operation_id": operation_id, "proposal_id": proposal_id,
                    "accepted_at": now,
                }
            proposal_refs = proposal.get("provenance") if isinstance(proposal.get("provenance"), list) else []
            if proposal_refs:
                state.setdefault("accepted_provenance", {})[accepted_card_id] = {
                    "version": accepted_version,
                    "provenance": [dict(ref) for ref in proposal_refs if isinstance(ref, dict)],
                }
            state["resolved"][issue_id] = {"action": action, "operation_id": operation_id, "created_at": now}
            operation = {"operation_id": operation_id, "action": action, "issue_id": issue_id,
                         "inputs": cards, "created_at": now, "source": request_snapshot["source"],
                         "request": request_snapshot,
                     "result": {"id": accepted_card_id, "version": accepted_version}}
            state["operations"][operation_id] = operation
            # Keep the JSON audit state and SQLite projection in one
            # recoverable transaction.  The fact file is restored below
            # if either persistence step fails.
            _audit_save_state(index_path, state)
            state_saved = True
            if rebuild_links:
                _db_rebuild_links(c)
                _db_sync_call_aggregates(c)
            c.commit()
        except Exception:
            try:
                c.rollback()
            finally:
                if state_saved:
                    _audit_restore_state(state_path, previous_state)
                if before_file is None:
                    try:
                        os.remove(target_path)
                    except FileNotFoundError:
                        pass
                else:
                    _audit_restore_state(target_path, before_file)
            raise
        impression.sweep_stale(c)
        _IDX.update({"data": None, "vocab": None, "df": None})
        _GRAPH["built"] = False; _GRAPH["nodes"] = {}; _GRAPH["edges"] = {}
        return {"ok": True, "idempotent": False, "operation": operation}
    except Exception:
        raise


def audit_apply(req):
    """Apply one human-confirmed audit action to the plugin-owned state file."""
    index_path = str(req.get("index") or "")
    factsdir = str(req.get("factsdir") or "")
    action = str(req.get("action") or "").strip().lower()
    operation_id = str(req.get("operation_id") or "").strip().lower()
    issue_id = str(req.get("issue_id") or "").strip()
    if action not in ("merge", "soft_delete", "keep", "accept"):
        return {"error": "invalid audit action"}
    try:
        operation_id = str(uuid.UUID(operation_id))
    except (ValueError, AttributeError):
        return {"error": "operation_id must be a UUID"}
    index_path = os.path.abspath(index_path)
    factsdir = os.path.abspath(factsdir)
    if (not index_path or not factsdir or not os.path.isdir(factsdir)
            or os.path.dirname(index_path) != factsdir
            or os.path.basename(index_path).lower() != "index.json"):
        return {"error": "plugin-owned facts directory unavailable"}
    c = dbside._db_for(index_path)
    if c is None:
        return {"error": "sidecar database unavailable"}
    state = _audit_load_state(index_path)
    request_snapshot = {
        "action": action, "issue_id": issue_id,
        "card_id": str(req.get("card_id") or ""),
        "versions": req.get("versions") if isinstance(req.get("versions"), dict) else {},
        "merged": str(req.get("merged") or "").strip(),
        "source": str(req.get("source") or "manual").strip()[:32] or "manual",
    }
    if request_snapshot["source"] not in ("manual", "automatic"):
        return {"error": "invalid audit source"}
    prior = state["operations"].get(operation_id)
    if prior:
        prior_snapshot = dict(prior.get("request") or {})
        prior_snapshot.setdefault("source", "manual")
        if prior_snapshot != request_snapshot:
            return {"error": "operation_id was already used for a different request", "conflict": True}
        c.execute("BEGIN IMMEDIATE")
        try:
            _audit_sync_rows(c, index_path, state)
            _db_rebuild_links(c)
            c.commit()
        except Exception:
            c.rollback()
            raise
        return {"ok": True, "idempotent": True, "operation": prior}
    versions = request_snapshot["versions"]
    targeted = ((issue_id.startswith("ai:") and len(versions) == 2)
                or (issue_id.startswith("ap:") and len(versions) == 1))
    if targeted:
        # 全量读取有效卡，仅复算当前候选：双卡重新确认身份，新事实检查
        # 全库冲突，均无须比较其他现有事实彼此。旧窗位置漂移也不受影响。
        candidates = audit_scan(factsdir, dup=0.85, near=0.5, max_facts=None,
                                index_path=index_path, only_issue=issue_id,
                                pair_ids=set(versions))
    else:
        candidates = audit_scan(factsdir, limit=400, dup=0.85, near=0.5, max_facts=1000, index_path=index_path)
    issue = next((row for row in candidates if row.get("issue_id") == issue_id), None)
    if issue is None and not targeted:
        # 旧窗候选：按页面缓存/请求窗口/更早窗口重新定位，而不是固定重扫最新 1000 条（S02）
        issue = _locate_issue(factsdir, index_path, issue_id, req)
    if issue is None:
        if action == "accept":
            proposal_id = next((str(proposal.get("id") or "")
                                for proposal in state.get("pending_proposals", [])
                                if isinstance(proposal, dict)
                                and "ap:" + str(proposal.get("id") or "") == issue_id), "")
            conflict_issue = next((row for row in candidates
                                   if row.get("proposal_conflict")
                                   and any(str(card.get("id") or "") == proposal_id
                                           for card in row.get("cards") or [])), None)
            if conflict_issue is not None:
                return {"error": "proposal conflicts with an existing fact; review or merge it first",
                        "conflict": True, "requires_action": "merge",
                        "audit_issue": conflict_issue}
        return {"error": "candidate is stale or no longer available", "conflict": True}
    cards = issue.get("cards") or []
    supplied_versions = request_snapshot["versions"]
    expected = {str(card.get("id")): str(card.get("version")) for card in cards}
    if supplied_versions != expected:
        return {"error": "fact card version changed; refresh the audit candidate", "conflict": True}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if action == "accept":
        return _accept_proposal(c, index_path, factsdir, state, issue, request_snapshot, operation_id, now,
                                review_candidates=candidates)
    provenance = []
    for card in cards:
        refs = card.get("provenance") or [{"source": card.get("source"), "line_number": card.get("line_number"), "line": card.get("line")}]
        provenance.extend(ref for ref in refs if isinstance(ref, dict))
    unique_provenance = []
    seen_refs = set()
    for ref in provenance:
        preserved = dict(ref)
        key = json.dumps(preserved, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in seen_refs:
            seen_refs.add(key)
            unique_provenance.append(preserved)
    if action == "keep":
        _audit_consume_proposals(state, cards, operation_id, now, action)
        operation = {"operation_id": operation_id, "action": action, "issue_id": issue_id,
                     "inputs": cards, "created_at": now, "source": request_snapshot["source"],
                     "request": request_snapshot}
        state["resolved"][issue_id] = {"action": action, "operation_id": operation_id, "created_at": now}
    elif action == "soft_delete":
        target = str(req.get("card_id") or "")
        if target not in expected:
            return {"error": "soft-delete target is not part of this candidate"}
        state["retired"][target] = {"version": expected[target], "reason": "soft_delete",
                                    "operation_id": operation_id, "created_at": now}
        if any(card.get("proposal") and str(card.get("id") or "") == target
               for card in cards):
            _audit_consume_proposals(state, cards, operation_id, now, action)
        state["resolved"][issue_id] = {"action": action, "operation_id": operation_id, "created_at": now}
        operation = {"operation_id": operation_id, "action": action, "issue_id": issue_id,
                     "inputs": cards, "target": target, "created_at": now,
                     "source": request_snapshot["source"], "request": request_snapshot}
    else:
        if not issue.get("merge_allowed"):
            return {"error": "conflicting or differently tagged facts cannot be merged"}
        merged_line = request_snapshot["merged"]
        parsed = _parse_fact(merged_line)
        if not parsed or len(merged_line) > 2000:
            return {"error": "merged fact must use the date | person | topic | fact format"}
        if any(parsed[key] != cards[0].get(key) for key in ("who", "topic")):
            return {"error": "merged fact must keep the confirmed person and topic"}
        card_id = "merged:" + operation_id.replace("-", "")
        ref_labels = []
        for ref in unique_provenance:
            source = str(ref.get("source") or "出处未知")
            if ref.get("line_number"):
                ref_labels.append("%s#L%s" % (source, ref["line_number"]))
            elif ref.get("offset") is not None:
                ref_labels.append("%s@%s" % (source, ref.get("offset")))
            else:
                ref_labels.append(source)
        ref_text = "；".join(ref_labels)
        body = merged_line + ("\n  出处：" + ref_text if ref_text else "")
        version = hashlib.sha256(body.encode("utf-8")).hexdigest()
        merged_record = {
            "id": card_id, "version": version, "line": merged_line, "body": body,
            "source": "facts/合并事实/" + operation_id + ".md", "provenance": unique_provenance,
            "created_at": now,
        }
        state["merged"].append(merged_record)
        for card in cards:
            state["retired"][str(card["id"])] = {
                "version": str(card["version"]), "reason": "merged", "operation_id": operation_id,
                "replacement_id": card_id, "created_at": now,
            }
        # A proposal has no source-file row to retire.  Consume it from the
        # pending queue when the human chooses merge, otherwise it would keep
        # reappearing as a second candidate after the merged card is indexed.
        _audit_consume_proposals(state, cards, operation_id, now, action)
        state["resolved"][issue_id] = {"action": action, "operation_id": operation_id, "created_at": now}
        operation = {"operation_id": operation_id, "action": action, "issue_id": issue_id,
                     "inputs": cards, "result": merged_record, "created_at": now,
                     "source": request_snapshot["source"], "request": request_snapshot}
    state["operations"][operation_id] = operation
    state_path = os.path.join(os.path.dirname(index_path), "audit_state.json")
    try:
        with open(state_path, "rb") as fh:
            previous_state = fh.read()
    except FileNotFoundError:
        previous_state = None
    c.execute("BEGIN IMMEDIATE")
    state_saved = False
    try:
        _audit_sync_rows(c, index_path, state)
        _db_rebuild_links(c)
        _db_sync_call_aggregates(c)
        # Persist after projection succeeds. If SQLite refuses its commit,
        # restore the previous JSON snapshot so the next scan cannot observe
        # an operation whose database transaction rolled back.
        _audit_save_state(index_path, state)
        state_saved = True
        c.commit()
    except Exception:
        try:
            c.rollback()
        finally:
            if state_saved:
                _audit_restore_state(state_path, previous_state)
        raise
    impression.sweep_stale(c)
    _IDX.update({"data": None, "vocab": None, "df": None})
    _GRAPH["built"] = False
    _GRAPH["nodes"] = {}
    _GRAPH["edges"] = {}
    return {"ok": True, "idempotent": False, "operation": operation}


def audit_history(req):
    """Return durable audit operations for the management page."""
    try:
        index_path, _ = _audit_plugin_paths(req)
    except ValueError as exc:
        return {"error": str(exc)}
    state = _audit_load_state(index_path)
    try:
        limit = max(1, min(200, int(req.get("limit") or 100)))
    except (TypeError, ValueError):
        limit = 100
    rows = []
    for operation_id, operation in state.get("operations", {}).items():
        if not isinstance(operation, dict):
            continue
        row = {
            "operation_id": str(operation.get("operation_id") or operation_id),
            "action": str(operation.get("action") or ""),
            "source": str(operation.get("source") or operation.get("request", {}).get("source") or "manual"),
            "issue_id": str(operation.get("issue_id") or ""),
            "created_at": str(operation.get("created_at") or ""),
            "status": "undone" if operation.get("undone_at") else "applied",
            "undone_at": str(operation.get("undone_at") or ""),
            "target": str(operation.get("target") or ""),
            "result_id": str((operation.get("result") or {}).get("id") or ""),
            "input_count": len(operation.get("inputs") or []),
            "channel": "audit",
        }
        rows.append(row)
    # 人工编辑/软删同样是"已应用的业务操作"，必须出现在同一份历史里并允许撤回，
    # 否则页面只有落盘记录、没有用户可用的撤回入口（复核第四轮 §6）。
    try:
        fact_state = fact_ops.load_ops(index_path)
    except RuntimeError as exc:
        _warn("read fact operations for audit history", exc)
        fact_state = {"ops": []}
    for op in fact_state.get("ops") or []:
        rows.append({
            "operation_id": str(op.get("op_id") or ""),
            "action": "fact_edit" if op.get("action") == "edit" else "fact_retire",
            "source": str(op.get("source") or "manual"),
            "issue_id": "",
            "created_at": str(op.get("created_at") or ""),
            "status": "undone" if op.get("undone_at") else "applied",
            "undone_at": str(op.get("undone_at") or ""),
            "target": str(op.get("fact_id") or ""),
            "result_id": "",
            "input_count": 1,
            "channel": "fact",
        })
    rows.sort(key=lambda row: (row["created_at"], row["operation_id"]), reverse=True)
    return {"ok": True, "records": rows[:limit], "count": len(rows)}


def _normalized_fact(card):
    return re.sub(r"\s+", "", str(card.get("fact") or ""))


def _same_content_pair(cards):
    """自动合并只接受规范化后完全相同的正文。

    相似度只能说明"读起来像"，不能证明两条主张等价：双方都含某个否定词时，
    "喜欢" 与 "不喜欢" 的相似度仍有 0.93（5.1.2 复核 S01）。这里不再做语义判断，
    只把自动合并收窄到可证明相同的文本；近似对仍作为人工候选展示。
    """
    cards = list(cards or [])
    return (len(cards) == 2
            and _normalized_fact(cards[0]) == _normalized_fact(cards[1])
            # 跨日期的相同正文可能真是不同日子发生的事，自动合并只处理同一天
            and str(cards[0].get("date") or "") == str(cards[1].get("date") or ""))


def audit_auto(req):
    """Automatically merge only high-confidence duplicate candidates.

    Near-duplicates and conflicts remain visible to the human audit page. Every
    automatic merge goes through audit_apply, so versions, provenance,
    rollback and idempotency use the same transaction path as manual actions.
    """
    try:
        index_path, factsdir = _audit_plugin_paths(req)
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        limit = max(1, min(100, int(req.get("limit") or 20)))
    except (TypeError, ValueError):
        limit = 20
    applied = []
    warning = ""
    coverage = {}
    for _ in range(limit):
        state = _audit_load_state(index_path)
        candidates = audit_scan(factsdir, limit=400, dup=0.85, near=0.5,
                                max_facts=1000, index_path=index_path, coverage=coverage)
        issue = next((row for row in candidates
                      if row.get("kind") == "重复" and row.get("merge_allowed")
                      and not row.get("proposal")
                      and not any(card.get("proposal") for card in row.get("cards") or [])
                      and _same_content_pair(row.get("cards"))
                      and str(row.get("issue_id") or "") not in state.get("auto_suppressed", {})), None)
        if issue is None:
            break
        cards = issue.get("cards") or []
        versions = {str(card.get("id")): str(card.get("version")) for card in cards}
        result = audit_apply({
            "index": index_path, "factsdir": factsdir, "action": "merge",
            "operation_id": str(uuid.uuid4()), "issue_id": str(issue.get("issue_id") or ""),
            "versions": versions, "merged": str(issue.get("suggestion") or ""),
            "source": "automatic",
        })
        if not result.get("ok"):
            warning = str(result.get("error") or "automatic audit stopped because a candidate changed")
            break
        applied.append(result.get("operation") or {})
    history = audit_history({"index": index_path, "factsdir": factsdir, "limit": 200})
    remaining_coverage = {}
    remaining = audit_scan(factsdir, limit=100, dup=0.85, near=0.5,
                           max_facts=1000, index_path=index_path, coverage=remaining_coverage)
    return {"ok": True, "applied": len(applied), "operations": applied,
            "records": history.get("records", []), "pending": len(remaining),
            "coverage": remaining_coverage, "warning": warning}


def _audit_restore_proposal_inputs(state, operation, operation_id, now):
    """Restore proposal cards consumed by a merge or soft-delete operation."""
    pending_ids = {str(item.get("id") or "") for item in state.get("pending_proposals", [])
                   if isinstance(item, dict)}
    for card in operation.get("inputs") or []:
        if not card.get("proposal"):
            continue
        proposal_id = str(card.get("id") or "")
        if not proposal_id or proposal_id in pending_ids:
            continue
        line = str(card.get("line") or "").strip()
        if not line:
            continue
        line_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
        target = os.path.basename(str(card.get("source") or card.get("file") or ""))
        if not target.startswith("事实_") or not target.endswith(".md"):
            target = "事实_撤回提案.md"
        state.setdefault("pending_proposals", []).append({
            "id": proposal_id,
            "version": str(card.get("version") or line_hash),
            "line": line,
            "line_hash": line_hash,
            "target": target,
            "provenance": [dict(ref) for ref in (card.get("provenance") or []) if isinstance(ref, dict)],
            "mark_source": str(card.get("mark_source") or ""),
            "created_at": now,
        })
        pending_ids.add(proposal_id)
        mark_source = str(card.get("mark_source") or "")
        accepted = state.get("accepted_sources", {}).get(mark_source)
        if mark_source and isinstance(accepted, dict) and str(accepted.get("operation_id") or "") == operation_id:
            state["accepted_sources"].pop(mark_source, None)


def audit_undo(req):
    """Undo a merge or soft-delete operation without touching official files."""
    try:
        index_path, factsdir = _audit_plugin_paths(req)
    except ValueError as exc:
        return {"error": str(exc)}
    try:
        operation_id = str(uuid.UUID(str(req.get("operation_id") or "")))
    except (ValueError, AttributeError):
        return {"error": "operation_id must be a UUID"}
    c = dbside._db_for(index_path)
    if c is None:
        return {"error": "sidecar database unavailable"}
    state = _audit_load_state(index_path)
    operation = state.get("operations", {}).get(operation_id)
    if not isinstance(operation, dict):
        return {"error": "audit operation not found", "conflict": True}
    if operation.get("undone_at"):
        return {"ok": True, "idempotent": True, "operation": operation}
    action = str(operation.get("action") or "")
    if action not in ("merge", "soft_delete"):
        return {"error": "only merge and soft-delete operations can be undone"}
    if action == "merge":
        result_id = str((operation.get("result") or {}).get("id") or "")
        dependent = next((other for other_id, other in state.get("operations", {}).items()
                          if other_id != operation_id and isinstance(other, dict)
                          and not other.get("undone_at") and other.get("action") == "merge"
                          and any(str(card.get("id") or "") == result_id
                                  for card in other.get("inputs") or [])), None)
        if dependent is not None:
            return {"error": "this merge is used by another applied merge; undo the later merge first",
                    "conflict": True,
                    "dependent_operation_id": str(dependent.get("operation_id") or "")}
    previous_state_path = os.path.join(factsdir, "audit_state.json")
    try:
        with open(previous_state_path, "rb") as fh:
            previous_state = fh.read()
    except FileNotFoundError:
        previous_state = None
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    issue_id = str(operation.get("issue_id") or "")
    try:
        c.execute("BEGIN IMMEDIATE")
        if action == "merge":
            result = operation.get("result") or {}
            result_id = str(result.get("id") or "")
            state["merged"] = [item for item in state.get("merged", [])
                                if not (isinstance(item, dict) and str(item.get("id") or "") == result_id)]
            if result_id:
                c.execute("DELETE FROM entries WHERE id=?", (result_id,))
            for card in operation.get("inputs") or []:
                card_id = str(card.get("id") or "")
                retired = state.get("retired", {}).get(card_id)
                if isinstance(retired, dict) and str(retired.get("operation_id") or "") == operation_id:
                    state["retired"].pop(card_id, None)
                entry_id = str(card.get("entry_id") or "")
                if entry_id:
                    c.execute("UPDATE entries SET retired=0,retired_at=NULL WHERE id=?", (entry_id,))
            _audit_restore_proposal_inputs(state, operation, operation_id, now)
        else:
            target = str(operation.get("target") or "")
            retired = state.get("retired", {}).get(target)
            if isinstance(retired, dict) and str(retired.get("operation_id") or "") == operation_id:
                state["retired"].pop(target, None)
            entry_id = next((str(card.get("entry_id") or "") for card in operation.get("inputs") or []
                             if str(card.get("id") or "") == target), "")
            if entry_id:
                c.execute("UPDATE entries SET retired=0,retired_at=NULL WHERE id=?", (entry_id,))
            elif target:
                # Virtual merged cards have no source-file entry_id. Restore
                # their SQLite projection explicitly to keep browse and audit
                # state in sync after undoing a soft-delete.
                c.execute("UPDATE entries SET retired=0,retired_at=NULL WHERE id=?", (target,))
            _audit_restore_proposal_inputs(state, operation, operation_id, now)
        state.get("resolved", {}).pop(issue_id, None)
        state.setdefault("auto_suppressed", {})[issue_id] = {
            "operation_id": operation_id, "undone_at": now,
            "reason": "manual undo; leave candidate for human review",
        }
        operation["undone_at"] = now
        operation["status"] = "undone"
        _audit_save_state(index_path, state)
        _audit_sync_rows(c, index_path, state)
        _db_rebuild_links(c)
        _db_sync_call_aggregates(c)
        c.commit()
    except Exception:
        c.rollback()
        _audit_restore_state(previous_state_path, previous_state)
        raise
    impression.sweep_stale(c)
    _IDX.update({"data": None, "vocab": None, "df": None})
    _GRAPH["built"] = False; _GRAPH["nodes"] = {}; _GRAPH["edges"] = {}
    return {"ok": True, "idempotent": False, "operation": operation}


def _audit_sync_rows(c, index_path, state, maintenance_state=None):
    """Apply durable audit state and manual fact operations as an entries projection.

    ★唯一真相是 `fact_ops.projection()`：事实文件 / 合并卡 + 按顺序重放的人工操作。
      本函数只负责"投影 → entries"落库，不再自己从文件推导正文。
      这样审计侧与检索侧看到的是同一张卡（复核第四轮 P1-1 / P1-2）。
    """
    cards, conflicts = fact_ops.projection(index_path)
    by_card_id = {card["card_id"]: card for card in cards.values()}
    changed = False

    # 1) audit_state 的软删：只有版本仍匹配才生效（正文变了就是另一张卡）
    retired_entries = set()
    for card_id, retired in state.get("retired", {}).items():
        card = by_card_id.get(card_id)
        if card is None:
            continue
        if card["audit_version"] != str(retired.get("version") or ""):
            continue
        retired_entries.add(card["entry_id"])
        card["retired_at"] = str(retired.get("created_at") or card["retired_at"] or "")

    # 2) 合并卡：虚拟卡不在事实文件里，需要先落一行，正文与状态交给第 3 步统一对齐
    for merged in state.get("merged", []):
        if not isinstance(merged, dict):
            continue
        body = str(merged.get("body") or merged.get("line") or "")
        merged_id = str(merged.get("id") or "")
        if not merged_id or not _parse_fact(body):
            continue
        if c.execute("SELECT 1 FROM entries WHERE id=?", (merged_id,)).fetchone() is not None:
            continue
        parsed = _parse_fact(body)
        tags = _tag(body)
        if parsed["who"] and parsed["who"] not in tags["people"]:
            tags["people"].append(parsed["who"])
        if parsed["topic"] and parsed["topic"] not in tags["topics"]:
            tags["topics"].append(parsed["topic"])
        entry = {
            "id": merged_id, "anchor": hashlib.sha1(body[:80].encode("utf-8")).hexdigest()[:12],
            "source": merged["source"], "time": fact_time(parsed["date"])[1] or str(parsed["date"]) + "T12:00:00",
            "date": fact_time(parsed["date"])[0] or parsed["date"],
            "period": "日", "hour": 12, "head": parsed["topic"][:60],
            "summary": parsed["fact"][:300], "text": body, "people": tags["people"],
            "topics": tags["topics"], "things": tags["things"], "retired": False,
            "_hash": hashlib.sha1(re.sub(r"\s+", "", body).encode("utf-8")).hexdigest()[:16],
        }
        if not _db_upsert(c, entry):
            raise RuntimeError("failed to index merged fact")
        changed = True

    # 3) 投影对齐：人工编辑生效/撤回、软删生效/撤回、审计软删合并到一起判断
    applied, replay_conflicts = fact_ops.replay_rows(c, index_path, retired_entries, cards)
    for conflict in list(conflicts) + list(replay_conflicts):
        _warn("replay fact operation (%s)" % (conflict.get("reason") or "conflict"),
              RuntimeError(str(conflict.get("fact_id") or "")))
    # Every write-side projection must preserve maintenance decisions too.
    import maintenance
    maintained = maintenance.apply_state(c, os.path.dirname(os.path.abspath(index_path)), maintenance_state)
    return bool(changed or applied or any(maintained.get(k) for k in ("retired", "purged", "merged")))


def _audit_sync_db(index_path, state):
    """Apply the durable state as a projection; db_build also reapplies it after rebuild."""
    c = dbside._db_for(index_path)
    if c is None:
        raise RuntimeError("sidecar database unavailable")
    try:
        c.execute("BEGIN IMMEDIATE")
        changed = _audit_sync_rows(c, index_path, state)
        if changed:
            _db_rebuild_links(c)
        c.commit()
    except Exception:
        c.rollback()
        raise
    impression.sweep_stale(c)
    _IDX.update({"data": None, "vocab": None, "df": None})
    _GRAPH["built"] = False
    _GRAPH["nodes"] = {}
    _GRAPH["edges"] = {}


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
