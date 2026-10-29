"""调用成功回执、滚动 7 日窗口与采集缺口的统计写入。"""

import re
import datetime
import hashlib
import sqlite3


# >>> WP1 导入（自动生成，勿手改）
import dbside
import recall_policy
from dbside import _is_fact_source
from state import _warn
# <<< WP1 导入

def _db_stats_start(c):
    try:
        row = c.execute("SELECT value FROM call_stats_meta WHERE key='started_at'").fetchone()
        return str(row[0] or "") if row else ""
    except sqlite3.Error:
        # 旧库没有统计表属预期降级，返回空起点
        return ""


def db_record_calls(req):
    """Persist an idempotent acknowledgement from the plugin return boundary."""
    c = dbside._db_ready(req)
    if c is None:
        return {"recorded": 0, "error": "statistics sidecar database unavailable"}
    logical_id = str(req.get("call_id") or "").strip()[:128]
    channel = str(req.get("call_channel") or "").strip()[:32]
    refs = req.get("facts")
    if not logical_id or channel not in ("automatic", "memrecall", "memwhen") or not isinstance(refs, list):
        return {"recorded": 0, "error": "invalid call acknowledgement"}
    called_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    gap_start = str(req.get("gap_start") or "").strip()
    recorded = 0
    accepted = []
    try:
        c.execute("SAVEPOINT record_calls")
        for ref in refs[:200]:
            if not isinstance(ref, dict):
                continue
            fact_id = str(ref.get("id") or "").strip()[:1024]
            version = str(ref.get("version") or "").strip()[:128]
            if not fact_id or not version:
                continue
            current = c.execute("SELECT text_hash,body,summary,retired,source FROM entries WHERE id=?", (fact_id,)).fetchone()
            if current is None or bool(current["retired"]) or not _is_fact_source(current["source"]):
                continue
            current_version = str(current["text_hash"] or "")
            if not current_version:
                norm = re.sub(r"\s+", "", str(current["body"] or current["summary"] or ""))
                current_version = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]
            if current_version != version:
                continue
            cur = c.execute(
                "INSERT OR IGNORE INTO call_events(logical_id,fact_id,fact_version,channel,called_at) VALUES(?,?,?,?,?)",
                (logical_id, fact_id, version, channel, called_at),
            )
            if cur.rowcount:
                recorded += 1
                accepted.append(ref)
        if gap_start:
            try:
                parsed_gap = datetime.datetime.fromisoformat(gap_start.replace("Z", "+00:00"))
                if parsed_gap.tzinfo is None:
                    parsed_gap = parsed_gap.replace(tzinfo=datetime.timezone.utc)
                normalized_gap = parsed_gap.astimezone(datetime.timezone.utc).isoformat()
                c.execute("INSERT OR IGNORE INTO call_stats_gaps(started_at,ended_at) VALUES(?,?)",
                          (normalized_gap, called_at))
            except (TypeError, ValueError):
                pass
        if channel == "automatic" and accepted:
            recall_policy.note_injected(recall_policy.session_of(req), accepted,
                                        req.get("window_gen"))
        c.execute("RELEASE SAVEPOINT record_calls")
        c.commit()
        return {"recorded": recorded}
    except Exception as e:
        try:
            c.execute("ROLLBACK TO SAVEPOINT record_calls")
            c.execute("RELEASE SAVEPOINT record_calls")
            c.commit()
        except Exception as rollback_error:
            # 回滚失败不覆盖原始失败，但必须留痕，否则事务状态无从判断
            _warn("rollback call statistics savepoint", rollback_error)
        return {"recorded": 0, "error": "%s: %s" % (type(e).__name__, e)}


def _fact_call_stats(c, fact_id, now=None, gap_open_since=""):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    cutoff = (now - datetime.timedelta(days=7)).isoformat()
    started = _db_stats_start(c)
    try:
        row = c.execute(
            "SELECT COUNT(*) AS total,"
            "SUM(CASE WHEN channel='automatic' THEN 1 ELSE 0 END) AS automatic_total,"
            "SUM(CASE WHEN channel!='automatic' THEN 1 ELSE 0 END) AS tool_total,"
            "SUM(CASE WHEN called_at>? THEN 1 ELSE 0 END) AS recent,"
            "MAX(called_at) AS last_called,"
            "(SELECT channel FROM call_events WHERE fact_id=? "
            "ORDER BY called_at DESC,logical_id DESC LIMIT 1) AS last_channel "
            "FROM call_events WHERE fact_id=?", (cutoff, str(fact_id), str(fact_id))).fetchone()
        complete = False
        if started:
            try:
                started_dt = datetime.datetime.fromisoformat(started.replace("Z", "+00:00"))
                complete = started_dt <= now - datetime.timedelta(days=7)
            except (TypeError, ValueError):
                complete = False
        gap_rows = c.execute(
            "SELECT started_at,ended_at FROM call_stats_gaps WHERE ended_at>? AND started_at<?",
            (cutoff, now.isoformat())).fetchall()
        has_gap = bool(gap_rows)
        if gap_open_since:
            try:
                open_dt = datetime.datetime.fromisoformat(str(gap_open_since).replace("Z", "+00:00"))
                if open_dt.tzinfo is None:
                    open_dt = open_dt.replace(tzinfo=datetime.timezone.utc)
                has_gap = has_gap or open_dt.astimezone(datetime.timezone.utc) < now
            except (TypeError, ValueError):
                has_gap = True
        complete = complete and not has_gap
        recent = int(row["recent"] or 0)
        return {
            "statistics_available": True,
            "call_count": int(row["total"] or 0),
            "automatic_call_count": int(row["automatic_total"] or 0),
            "tool_call_count": int(row["tool_total"] or 0),
            "calls_7d": recent if complete else None,
            "calls_7d_observed": recent,
            "statistics_complete": complete,
            "statistics_start_time": started,
            "statistics_gap_count_7d": len(gap_rows) + (1 if gap_open_since else 0),
            "statistics_gap_open": bool(gap_open_since),
            "last_called_at": str(row["last_called"] or ""),
            "last_called_channel": str(row["last_channel"] or ""),
        }
    except (sqlite3.Error, KeyError, IndexError) as e:
        _warn("read call statistics: %s" % fact_id, e)
        return {
            "statistics_available": False,
            "call_count": None, "automatic_call_count": None, "tool_call_count": None,
            "calls_7d": None, "calls_7d_observed": 0,
            "statistics_complete": False, "statistics_start_time": started,
            "statistics_gap_count_7d": 0, "statistics_gap_open": bool(gap_open_since),
            "last_called_at": "", "last_called_channel": "",
        }
