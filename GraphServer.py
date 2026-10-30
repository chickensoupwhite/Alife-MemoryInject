"""Loopback-only companion server for the standalone memory graph page.

Security boundary:
  * listen on 127.0.0.1 only;
  * the SQLite index is fixed at startup (--index / ALIFE_MEMORY_GRAPH_INDEX),
    never taken from an HTTP query parameter;
  * graph queries are read-only; the only write endpoint applies human-confirmed
    audit actions to the plugin-owned audit state beside the configured index.
    Source paths are display data only.

Usage:
  python GraphServer.py --index <.../index.json> [--port 18763] [--no-model]
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE = Path(__file__).resolve().parent
HOST = "127.0.0.1"


def _arg_value(flag: str, default: str = "") -> str:
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


INDEX = _arg_value("--index", os.environ.get("ALIFE_MEMORY_GRAPH_INDEX", ""))
if INDEX:
    INDEX = os.path.abspath(INDEX)
    if os.path.isdir(INDEX):
        INDEX = os.path.join(INDEX, "index.json")
INDEX_HASH = (
    hashlib.sha256(os.path.normcase(INDEX).encode("utf-8")).hexdigest()[:16] if INDEX else ""
)
try:
    PORT = int(_arg_value("--port", os.environ.get("ALIFE_MEMORY_GRAPH_PORT", "18763")))
except ValueError:
    PORT = 18763
NO_MODEL = "--no-model" in sys.argv[1:] or os.environ.get("ALIFE_MEMORY_GRAPH_NO_MODEL") == "1"
WATCH_STDIN = os.environ.get("ALIFE_MEMORY_GRAPH_PARENT_STDIN") == "1"

_lock = threading.Lock()
_proc = None
_ready = False
_current_index = ""
_out_queue = None
_reader = None


def _log(message: str) -> None:
    try:
        sys.stderr.write("[graph] %s\n" % message)
        sys.stderr.flush()
    except OSError:
        pass


def _read_sidecar(proc, outq) -> None:
    try:
        for line in proc.stdout:
            outq.put(line)
    except (OSError, ValueError):
        pass
    finally:
        outq.put(None)


def _stop_sidecar() -> None:
    global _proc, _ready, _current_index, _out_queue, _reader
    proc = _proc
    _proc = None
    _ready = False
    _current_index = ""
    _out_queue = None
    _reader = None
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.write('{"id":0,"fn":"__shutdown__"}\n')
            proc.stdin.flush()
    except OSError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


atexit.register(_stop_sidecar)


def _spawn_sidecar() -> None:
    global _proc, _ready, _current_index, _out_queue, _reader
    cmd = [sys.executable, str(BASE / "sidecar_embed.py")]
    if NO_MODEL:
        cmd.append("--no-model")
    _proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        cwd=str(BASE),
        bufsize=1,
    )
    _out_queue = queue.Queue()
    _reader = threading.Thread(target=_read_sidecar, args=(_proc, _out_queue), daemon=True)
    _reader.start()
    _ready = False
    _current_index = ""


def _read_line(timeout: float, what: str) -> str:
    try:
        line = _out_queue.get(timeout=max(0.05, timeout))
    except queue.Empty:
        raise TimeoutError("sidecar %s timeout" % what)
    if line is None:
        raise RuntimeError("sidecar exited during %s" % what)
    return line


def _ensure_sidecar(index: str):
    global _ready, _current_index
    if _proc is None or _proc.poll() is not None:
        _spawn_sidecar()
    if not _ready:
        obj = json.loads(_read_line(20, "ready"))
        if obj.get("fn") != "ready" or not obj.get("ok"):
            raise RuntimeError("sidecar handshake failed")
        _ready = True
    if index != _current_index:
        _proc.stdin.write(
            json.dumps({"id": 0, "fn": "warm", "index": index, "readonly": True}, ensure_ascii=False) + "\n"
        )
        _proc.stdin.flush()
        deadline = time.monotonic() + 20
        while True:
            obj = json.loads(_read_line(deadline - time.monotonic(), "warm"))
            if obj.get("id") == 0:
                if not obj.get("ok"):
                    raise RuntimeError("sidecar warm failed")
                break
        _current_index = index
    return _proc


def _do_request(req):
    proc = _ensure_sidecar(req["index"])
    proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    deadline = time.monotonic() + 30
    while True:
        obj = json.loads(_read_line(deadline - time.monotonic(), "request"))
        if obj.get("id") == req["id"]:
            return obj


def _request_sidecar(req):
    """Serialize requests; retry once if the sidecar died between requests."""
    last = None
    for _attempt in range(2):
        try:
            return _do_request(req)
        except Exception as exc:
            last = exc
        _stop_sidecar()
    raise last or RuntimeError("sidecar unavailable")


def _param(query: dict, key: str, default: str = "") -> str:
    return (query.get(key, [default])[0] or default).strip()


def _int_param(query: dict, key: str, default: int, lo: int, hi: int) -> int:
    raw = _param(query, key, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(key)
    return max(lo, min(hi, value))


class Handler(BaseHTTPRequestHandler):
    server_version = "MemoryGraph/1.0"
    sys_version = ""

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body, ctype: str, csp: str = "") -> None:
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if csp:
                self.send_header("Content-Security-Policy", csp)
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False), "application/json; charset=utf-8")

    def do_OPTIONS(self):
        self._send(405, "method not allowed", "text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/impression/refresh":
            name = (parse_qs(parsed.query).get("name") or [""])[0]
            if not name.strip():
                return self._json(400, {"error": "name is required"})
            req = {"id": int.from_bytes(os.urandom(4), "big"), "fn": "impressionstale",
                   "index": INDEX, "name": name}
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {"error": "impression refresh timed out"})
            except Exception as e:
                _log("backend failure: %s: %s" % (type(e).__name__, e))
                return self._json(503, {"error": "impression service unavailable"})
            if not result or not result.get("ok"):
                return self._json(503, {"error": (result or {}).get("error", "refresh failed")})
            return self._json(200, {"ok": True, "name": name, "marked": bool(result.get("marked")),
                                    "note": "已标记过期，下次 AI 调用 mem_impression 时重新生成"})
        if parsed.path not in ("/audit", "/audit/auto", "/audit/undo", "/fact/edit",
                               "/fact/retire", "/fact/undo", "/entity/confirm",
                               "/entity/reject", "/entity/manage") or parsed.query:
            return self._json(404, {"error": "not found"})
        if not INDEX:
            return self._json(503, {"error": "graph index not configured"})
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            return self._json(400, {"error": "application/json is required"})
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json(400, {"error": "invalid Content-Length"})
        if length < 1 or length > 65536:
            return self._json(400, {"error": "request body must be between 1 and 65536 bytes"})
        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._json(400, {"error": "invalid JSON body"})
        if not isinstance(payload, dict):
            return self._json(400, {"error": "JSON body must be an object"})
        if parsed.path == '/entity/manage':
            if set(payload) - {'action', 'name', 'kind', 'target'} or any(not isinstance(value, str) for value in payload.values()):
                return self._json(400, {'error': 'invalid portrait management fields'})
            req = dict(payload, id=int.from_bytes(os.urandom(4), 'big'), fn='profilemanage',
                       index=INDEX, factsdir=os.path.dirname(INDEX))
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {'error': '画像操作超时，请刷新列表核对结果'})
            except Exception as exc:
                _log('portrait management failed: %s' % exc)
                return self._json(503, {'error': '画像管理暂时不可用'})
            return self._json(200 if result and result.get('ok') else 400,
                              result or {'error': '画像操作失败'})
        if parsed.path in ("/entity/confirm", "/entity/reject"):
            if set(payload) - {"name"} or not isinstance(payload.get("name"), str):
                return self._json(400, {"error": "name is required"})
            name = payload["name"].strip()
            if not name or len(name) > 64:
                return self._json(400, {"error": "invalid person name"})
            fn = "peopleconfirm" if parsed.path == "/entity/confirm" else "peoplereject"
            req = {"id": int.from_bytes(os.urandom(4), "big"), "fn": fn,
                   "index": INDEX, "factsdir": os.path.dirname(INDEX), "name": name}
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {"error": "person classification timed out"})
            except Exception as e:
                _log("backend failure: %s: %s" % (type(e).__name__, e))
                return self._json(503, {"error": "management service unavailable"})
            if not result or not result.get("ok"):
                return self._json(400, {"error": (result or {}).get("error", "classification failed")})
            return self._json(200, {"ok": True, "name": result.get("name", name),
                                    "confirmed": result.get("confirmed", []),
                                    "entries_reclassified": int(result.get("entries_reclassified", 0))})
        if parsed.path == "/audit/auto":
            if set(payload) - {"limit"}:
                return self._json(400, {"error": "unsupported automatic audit fields"})
            try:
                limit = int(payload.get("limit", 20))
            except (TypeError, ValueError):
                return self._json(400, {"error": "invalid audit limit"})
            if limit < 1 or limit > 100:
                return self._json(400, {"error": "invalid audit limit"})
            req = {"id": int.from_bytes(os.urandom(4), "big"), "fn": "auditauto",
                   "index": INDEX, "factsdir": os.path.dirname(INDEX), "limit": limit}
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {"error": "automatic audit timed out"})
            except Exception as e:
                _log("backend failure: %s: %s" % (type(e).__name__, e))
                return self._json(503, {"error": "management service unavailable"})
            if not result or not result.get("ok"):
                return self._json(503, {"error": result.get("error", "automatic audit failed")
                                        if isinstance(result, dict) else "automatic audit failed"})
            return self._json(200, {"ok": True, "applied": int(result.get("applied", 0)),
                                    "records": result.get("records", []),
                                    "coverage": result.get("coverage", {}),
                                    "warning": str(result.get("warning") or "")})
        if parsed.path == "/audit/undo":
            if set(payload) - {"operation_id"} or not isinstance(payload.get("operation_id"), str):
                return self._json(400, {"error": "operation_id is required"})
            try:
                operation_id = str(uuid.UUID(payload["operation_id"]))
            except (ValueError, AttributeError):
                return self._json(400, {"error": "operation_id must be a UUID"})
            req = {"id": int.from_bytes(os.urandom(4), "big"), "fn": "auditundo",
                   "index": INDEX, "factsdir": os.path.dirname(INDEX),
                   "operation_id": operation_id}
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {"error": "audit undo timed out"})
            except Exception as e:
                _log("backend failure: %s: %s" % (type(e).__name__, e))
                return self._json(503, {"error": "management service unavailable"})
            if not result or not result.get("ok"):
                if isinstance(result, dict) and result.get("conflict"):
                    return self._json(409, {"error": result.get("error", "audit operation changed"), "conflict": True})
                return self._json(400, {"error": result.get("error", "audit undo failed")
                                        if isinstance(result, dict) else "audit undo failed"})
            return self._json(200, {"ok": True, "idempotent": bool(result.get("idempotent", False)),
                                    "operation": result.get("operation", {})})
        if parsed.path == "/fact/undo":
            if set(payload) - {"op_id"}:
                return self._json(400, {"error": "unsupported fact undo fields"})
            try:
                op_id = str(uuid.UUID(str(payload.get("op_id") or "")))
            except (ValueError, AttributeError):
                return self._json(400, {"error": "op_id must be a UUID"})
            req = {"id": int.from_bytes(os.urandom(4), "big"), "fn": "factundo",
                   "index": INDEX, "factsdir": os.path.dirname(INDEX), "op_id": op_id}
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {"error": "fact undo timed out"})
            except Exception as e:
                _log("backend failure: %s: %s" % (type(e).__name__, e))
                return self._json(503, {"error": "management service unavailable"})
            if not result or not result.get("ok"):
                if isinstance(result, dict) and result.get("conflict"):
                    return self._json(409, {"error": result.get("error", "fact operation not found"),
                                            "conflict": True})
                if isinstance(result, dict) and result.get("error"):
                    return self._json(400, {"error": result["error"]})
                return self._json(503, {"error": "fact undo failed"})
            return self._json(200, {"ok": True,
                                    "idempotent": bool(result.get("idempotent", False)),
                                    "operation": result.get("operation", {})})
        if parsed.path in ("/fact/edit", "/fact/retire"):
            # 独立事实卡（没有审计候选）的人工编辑/软删。语义与 /audit 一致：
            # 必须带 op_id 做幂等、带 base_version 做版本校验，路径一律服务端定。
            action = "edit" if parsed.path == "/fact/edit" else "retire"
            if set(payload) - {"op_id", "fact_id", "base_version", "new_body"}:
                return self._json(400, {"error": "unsupported fact operation fields"})
            try:
                op_id = str(uuid.UUID(str(payload.get("op_id") or "")))
            except (ValueError, AttributeError):
                return self._json(400, {"error": "op_id must be a UUID"})
            fact_id = payload.get("fact_id")
            if not isinstance(fact_id, str) or not fact_id or len(fact_id) > 512:
                return self._json(400, {"error": "fact_id is required"})
            base_version = payload.get("base_version")
            if not isinstance(base_version, str) or not base_version or len(base_version) > 128:
                return self._json(400, {"error": "base_version is required"})
            new_body = payload.get("new_body", "")
            if not isinstance(new_body, str):
                return self._json(400, {"error": "invalid fact body"})
            if action == "retire" and new_body.strip():
                return self._json(400, {"error": "retire does not accept a new body"})
            req = {
                "id": int.from_bytes(os.urandom(4), "big"),
                "fn": "factapply",
                "index": INDEX,
                # 与 /audit 同样不接受客户端路径：只认插件自己的索引目录
                "factsdir": os.path.dirname(INDEX),
                "action": action,
                "op_id": op_id,
                "fact_id": fact_id,
                "base_version": base_version,
                "new_body": new_body,
                "source": "manual",
            }
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {"error": "fact operation timed out"})
            except Exception as e:
                _log("backend failure: %s: %s" % (type(e).__name__, e))
                return self._json(503, {"error": "management service unavailable"})
            if not result or not result.get("ok"):
                if isinstance(result, dict) and result.get("conflict"):
                    return self._json(409, {"error": result.get("error", "fact card changed"),
                                            "conflict": True})
                if isinstance(result, dict) and result.get("error"):
                    return self._json(400, {"error": result["error"]})
                return self._json(503, {"error": "fact operation failed"})
            return self._json(200, {
                "ok": True,
                "idempotent": bool(result.get("idempotent", False)),
                "operation": result.get("operation", {}),
            })
        allowed_fields = {"action", "operation_id", "issue_id", "card_id", "versions", "merged",
                          "offset", "max_facts"}
        if set(payload) - allowed_fields:
            return self._json(400, {"error": "unsupported audit fields"})
        action = payload.get("action")
        if action not in ("merge", "soft_delete", "keep", "accept"):
            return self._json(400, {"error": "invalid audit action"})
        if not isinstance(payload.get("operation_id"), str):
            return self._json(400, {"error": "operation_id must be a UUID"})
        try:
            operation_id = str(uuid.UUID(payload["operation_id"]))
        except (ValueError, AttributeError):
            return self._json(400, {"error": "operation_id must be a UUID"})
        issue_id = payload.get("issue_id")
        if not isinstance(issue_id, str) or not issue_id or len(issue_id) > 128:
            return self._json(400, {"error": "issue_id is required"})
        card_id = payload.get("card_id", "")
        merged = payload.get("merged", "")
        versions = payload.get("versions", {})
        if not isinstance(card_id, str) or len(card_id) > 128:
            return self._json(400, {"error": "invalid card_id"})
        if not isinstance(merged, str) or len(merged) > 2000:
            return self._json(400, {"error": "invalid merged fact"})
        if (not isinstance(versions, dict) or len(versions) > 4
                or any(not isinstance(key, str) or len(key) > 128
                       or not isinstance(value, str) or len(value) > 128
                       for key, value in versions.items())):
            return self._json(400, {"error": "invalid versions"})
        try:
            window_offset = int(payload.get("offset") or 0)
            window_max = int(payload.get("max_facts") or 400)
        except (TypeError, ValueError):
            return self._json(400, {"error": "invalid audit window"})
        if window_offset < 0 or window_offset > 1000000 or window_max < 1 or window_max > 2000:
            return self._json(400, {"error": "invalid audit window"})
        req = {
            "id": int.from_bytes(os.urandom(4), "big"),
            "fn": "auditapply",
            "index": INDEX,
            # Never trust a client-supplied index or facts directory. The
            # sidecar requires these to be the plugin's own index directory.
            "factsdir": os.path.dirname(INDEX),
            "action": action,
            "operation_id": operation_id,
            "issue_id": issue_id,
            "card_id": card_id,
            "versions": versions,
            "merged": merged,
            "source": "manual",
            "offset": window_offset,
            "max_facts": window_max,
        }
        try:
            with _lock:
                result = _request_sidecar(req)
        except TimeoutError:
            return self._json(504, {"error": "audit update timed out"})
        except Exception as e:
            _log("backend failure: %s: %s" % (type(e).__name__, e))
            return self._json(503, {"error": "management service unavailable"})
        if not result or not result.get("ok"):
            if isinstance(result, dict) and result.get("conflict"):
                payload = {"error": result.get("error", "audit candidate changed"),
                           "conflict": True}
                if result.get("requires_action"):
                    payload["requires_action"] = result["requires_action"]
                if isinstance(result.get("candidates"), list):
                    payload["candidates"] = result["candidates"]
                if isinstance(result.get("audit_issue"), dict):
                    payload["audit_issue"] = result["audit_issue"]
                return self._json(409, payload)
            if isinstance(result, dict) and result.get("error"):
                return self._json(400, {"error": result["error"]})
            return self._json(503, {"error": "audit update failed"})
        return self._json(200, {
            "ok": True,
            "idempotent": bool(result.get("idempotent", False)),
            "operation": result.get("operation", {}),
        })

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/organize/progress":
            if not INDEX:
                return self._json(503, {"error": "graph index not configured"})
            try:
                snapshot = Path(INDEX).parent / "organize_status.json"
                payload = json.loads(snapshot.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or not isinstance(payload.get("layers"), list) or not isinstance(payload.get("files"), list):
                    raise ValueError("invalid organize progress")
            except FileNotFoundError:
                return self._json(503, {"error": "来源进度尚未生成，请重载插件后从插件打开管理页。"})
            except (OSError, ValueError):
                return self._json(503, {"error": "来源进度暂时无法读取，请稍后刷新。"})
            return self._json(200, {"ok": True, **payload})
        if path == "/impression":
            name = (parse_qs(parsed.query).get("name") or [""])[0]
            if not name.strip():
                return self._json(400, {"error": "name is required"})
            req = {"id": int.from_bytes(os.urandom(4), "big"), "fn": "impression",
                   "index": INDEX, "name": name, "readonly": True}
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {"error": "impression request timed out"})
            except Exception as e:
                _log("backend failure: %s: %s" % (type(e).__name__, e))
                return self._json(503, {"error": "impression service unavailable"})
            if not result or not result.get("ok"):
                return self._json(503, {"error": (result or {}).get("error", "impression failed")})
            return self._json(200, result)
        if path == "/health":
            return self._json(200, {
                "ok": True,
                "service": "memory-graph",
                "index_hash": INDEX_HASH,
                "model": "disabled" if NO_MODEL else "auto",
                "version": 2,
            })
        if path in ("/stats", "/facts", "/source", "/audit", "/profile", "/entities"):
            if not INDEX:
                return self._json(503, {"error": "graph index not configured"})
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                request_id = int.from_bytes(os.urandom(4), "big")
                if path == "/stats":
                    req = {"id": request_id, "fn": "stats", "index": INDEX, "readonly": True}
                elif path == "/audit":
                    # factsRoot 与 index.json 同目录；不接受客户端传入的路径，
                    # 这样审计只能读取插件自己的事实卡侧车。
                    req = {
                        "id": request_id, "fn": "auditscan", "index": INDEX, "readonly": True,
                        "factsdir": os.path.dirname(INDEX),
                        "limit": _int_param(query, "limit", 30, 1, 100),
                        # 继续扫描必须真的换窗：只给 limit 会一直返回最新 400 条
                        "offset": _int_param(query, "offset", 0, 0, 1000000),
                        "max_facts": _int_param(query, "max_facts", 400, 1, 2000),
                        "dup": 0.85,
                        "near": 0.5,
                    }
                elif path == "/source":
                    source = _param(query, "source")[:512]
                    if not source:
                        return self._json(400, {"error": "source is required"})
                    req = {"id": request_id, "fn": "fetch", "source": source,
                           "limit": 3, "index": INDEX, "readonly": True}
                elif path == "/profile":
                    profile_name = _param(query, "name").strip()
                    profile_kind = _param(query, "kind", "auto").strip().lower()
                    profile_sort = _param(query, "sort", "name").strip().lower()
                    profile_usage = _param(query, "usage").strip().lower()
                    profile_importance = _param(query, "importance").strip().lower()
                    if not profile_name or len(profile_name) > 120:
                        return self._json(400, {"error": "profile name must be between 1 and 120 characters"})
                    if profile_kind not in ("auto", "person", "event", "object"):
                        return self._json(400, {"error": "invalid profile kind"})
                    if profile_sort not in ("name", "call_count", "calls_7d", "importance", "last_called"):
                        return self._json(400, {"error": "invalid profile sort option"})
                    if profile_usage not in ("", "used", "unused"):
                        return self._json(400, {"error": "invalid profile usage filter"})
                    if profile_importance not in ("", "known", "unset"):
                        return self._json(400, {"error": "invalid profile importance filter"})
                    req = {
                        "id": request_id, "fn": "profile", "index": INDEX, "readonly": True,
                        "name": profile_name,
                        "kind": profile_kind,
                        "offset": _int_param(query, "offset", 0, 0, 100000),
                        "limit": _int_param(query, "limit", 50, 1, 100),
                        "sort": profile_sort,
                        "usage": profile_usage,
                        "importance": profile_importance,
                    }
                elif path == "/entities":
                    entity_kind = _param(query, "kind", "person").strip().lower()
                    if entity_kind not in ("person", "event", "object", "pending", "shared"):
                        return self._json(400, {"error": "invalid entity kind"})
                    req = {
                        "id": request_id, "fn": "entities", "index": INDEX, "readonly": True,
                        "kind": entity_kind,
                        "offset": _int_param(query, "offset", 0, 0, 100000),
                        "limit": _int_param(query, "limit", 60, 1, 200),
                    }
                else:
                    facts_offset = _int_param(query, "offset", 0, 0, 100000)
                    facts_limit = _int_param(query, "limit", 50, 1, 200)
                    facts_sort = _param(query, "sort", "name")
                    facts_usage = _param(query, "usage")
                    facts_importance = _param(query, "importance")
                    if facts_sort not in ("name", "call_count", "calls_7d", "importance", "last_called"):
                        return self._json(400, {"error": "invalid sort option"})
                    if facts_usage not in ("", "used", "unused"):
                        return self._json(400, {"error": "invalid usage filter"})
                    if facts_importance not in ("", "known", "unset"):
                        return self._json(400, {"error": "invalid importance filter"})
                    req = {
                        "id": request_id, "fn": "browse", "index": INDEX, "readonly": True,
                        "facts_only": True,
                        "query": _param(query, "query")[:200],
                        "who": _param(query, "who")[:120],
                        "topic": _param(query, "topic")[:120],
                        "thing": _param(query, "thing")[:120],
                        "offset": facts_offset,
                        "limit": facts_limit,
                        "sort": facts_sort,
                        "usage": facts_usage,
                        "importance": facts_importance,
                    }
            except ValueError:
                return self._json(400, {"error": "invalid query parameters"})
            try:
                with _lock:
                    result = _request_sidecar(req)
            except TimeoutError:
                return self._json(504, {"error": "management request timed out"})
            except Exception as e:
                _log("backend failure: %s: %s" % (type(e).__name__, e))
                return self._json(503, {"error": "management service unavailable"})
            if not result or not result.get("ok"):
                return self._json(503, {"error": result.get("error", "index unavailable")
                                        if isinstance(result, dict) else "index unavailable"})
            if path == "/stats":
                return self._json(200, {
                    "count": result.get("count", 0),
                    "first": result.get("first", ""),
                    "last": result.get("last", ""),
                    "text": result.get("text", ""),
                })
            if path == "/audit":
                return self._json(200, {
                    "count": result.get("count", 0),
                    "issues": result.get("issues", []),
                    "history": result.get("history", []),
                    # 覆盖范围必须和候选一起返回，否则"没有候选"会被误读成全库没问题（R10）
                    "coverage": result.get("coverage", {}),
                })
            if path == "/source":
                return self._json(200, {"source": req["source"], "text": result.get("text", "")})
            if path == "/entities":
                return self._json(200, {
                    "kind": result.get("kind", req.get("kind", "person")),
                    "items": result.get("items", []),
                    "total": int(result.get("total", 0)),
                    "offset": int(result.get("offset", req.get("offset", 0))),
                    "truncated": bool(result.get("truncated", False)),
                    "has_more": bool(result.get("has_more", False)),
                    # 待分类／共同主体与人物总数同源，页面底部数量必须能对上（R04）
                    "pending": result.get("pending", []),
                    "pending_total": int(result.get("pending_total", 0)),
                    "shared": result.get("shared", []),
                    "shared_total": int(result.get("shared_total", 0)),
                    "note": str(result.get("note", "")),
                    "management": result.get("management", {}),
                })
            if path == "/profile":
                return self._json(200, {
                    "entity": result.get("entity", {}),
                    "available_kinds": result.get("available_kinds", []),
                    "requires_kind": bool(result.get("requires_kind", False)),
                    "found": bool(result.get("found", False)),
                    "total": int(result.get("total", 0)),
                    "offset": int(result.get("offset", req["offset"])),
                    "limit": req["limit"],
                    "has_more": bool(result.get("has_more", False)),
                    "relations": result.get("relations", []),
                    "relations_complete": bool(result.get("relations_complete", True)),
                    "summary": result.get("summary", {}),
                    "items": result.get("items", []),
                    "text": result.get("text", ""),
                })
            items = result.get("items", [])
            offset = result.get("offset", req["offset"])
            total = result.get("total", 0)
            return self._json(200, {
                "total": total,
                "offset": offset,
                "limit": req["limit"],
                "has_more": offset + len(items) < total,
                "items": items,
                "text": result.get("text", ""),
            })
        if path == "/":
            try:
                html = (BASE / "DetailPage.html").read_bytes()
            except OSError:
                return self._send(404, "detail page missing", "text/plain; charset=utf-8")
            return self._send(
                200, html, "text/html; charset=utf-8",
                "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; "
                "img-src data:; connect-src 'self'; base-uri 'none'; form-action 'none'",
            )
        if path == "/d3.v7.min.js":
            try:
                d3 = (BASE / "d3.v7.min.js").read_bytes()
            except OSError:
                return self._send(404, "D3 bundle missing", "text/plain; charset=utf-8")
            return self._send(200, d3, "text/javascript; charset=utf-8")
        if path != "/graph":
            return self._send(404, "not found", "text/plain; charset=utf-8")
        if not INDEX:
            return self._json(503, {"error": "graph index not configured"})
        query = parse_qs(parsed.query, keep_blank_values=True)
        try:
            q = _param(query, "query")[:200]
            expand = _param(query, "expand")[:512]
            top = _int_param(query, "top", 8, 1, 200)
            max_nodes = _int_param(query, "max_nodes", 100, 10, 250)
            offset = _int_param(query, "offset", 0, 0, 1000000)
            entry_offset = _int_param(query, "entry_offset", 0, 0, 1000000)
            entry_limit = _int_param(query, "entry_limit", 24, 1, 200)
            # relation_limit 只在调用方显式给出时才下发：不传时侧车沿用旧分页默认
            # （搜索路径 top、概览 30），避免改变既有前端行为。
            raw_limit = _param(query, "relation_limit", "")
            relation_limit = _int_param(query, "relation_limit", 30, 1, 200) if raw_limit else None
            include_entries = _param(
                query, "include_entries", "1" if q else "0"
            ).lower() in ("1", "true", "yes", "on")
        except ValueError:
            return self._json(400, {"error": "invalid query parameters"})
        req = {
            "id": int.from_bytes(os.urandom(4), "big"),
            "fn": "graphdata",
            "query": q,
            "expand": expand,
            "top": top,
            "max_nodes": max_nodes,
            "offset": offset,
            "entry_offset": entry_offset,
            "entry_limit": entry_limit,
            "include_entries": include_entries,
            "readonly": True,
            "index": INDEX,
        }
        if relation_limit is not None:
            req["relation_limit"] = relation_limit
        try:
            with _lock:
                result = _request_sidecar(req)
        except TimeoutError:
            return self._json(504, {"error": "graph request timed out"})
        except Exception as e:
            _log("backend failure: %s: %s" % (type(e).__name__, e))
            return self._json(503, {"error": "graph service unavailable"})
        if not result or not result.get("ok"):
            return self._json(503, {"error": result.get("error", "index unavailable") if isinstance(result, dict) else "index unavailable"})
        return self._json(200, {
            "nodes": result.get("nodes", []),
            "edges": result.get("edges", []),
            "truncated": bool(result.get("truncated", False)),
            "returned": int(result.get("returned", 0) or 0),
            "offset": int(result.get("offset", 0) or 0),
            "next_offset": int(result.get("next_offset", 0) or 0),
            "total_relations": int(result.get("total_relations", 0) or 0),
            "has_more": bool(result.get("has_more", False)),
            "entries_returned": int(result.get("entries_returned", 0) or 0),
            "entry_offset": int(result.get("entry_offset", 0) or 0),
            "total_entries": int(result.get("total_entries", 0) or 0),
            "entries_has_more": bool(result.get("entries_has_more", False)),
            "retired_hidden": int(result.get("retired_hidden", 0) or 0),
        })


class GraphHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # Windows: SO_EXCLUSIVEADDRUSE and SO_REUSEADDR cannot be combined.
    allow_reuse_address = False

    def server_bind(self):
        # Windows: refuse to share the port with another process.
        if os.name == "nt":
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            except OSError:
                pass
        super().server_bind()


def _probe_local_service():
    try:
        with urllib.request.urlopen(
            "http://%s:%s/health" % (HOST, PORT), timeout=0.6
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def main() -> int:
    if not (1 <= PORT <= 65535):
        _log("invalid port: %s" % PORT)
        return 2
    if not INDEX:
        _log("no index configured; set ALIFE_MEMORY_GRAPH_INDEX or --index")
    for required in (BASE / "sidecar_embed.py", BASE / "DetailPage.html", BASE / "d3.v7.min.js"):
        if not required.is_file():
            _log("required graph file missing: %s" % required.name)
            return 2
    if INDEX and not os.path.isfile(INDEX):
        db_candidate = os.path.join(os.path.dirname(INDEX), "memory.db")
        if not os.path.isfile(db_candidate):
            _log("configured index file missing and no memory.db fallback: %s" % INDEX)
            return 2
        _log("index.json missing; graph will use existing memory.db only")
    existing = _probe_local_service()
    if isinstance(existing, dict) and existing.get("service") == "memory-graph":
        try:
            with urllib.request.urlopen("http://%s:%s/" % (HOST, PORT), timeout=0.6) as page:
                page_ok = page.status == 200 and "text/html" in page.headers.get("Content-Type", "")
        except (OSError, ValueError):
            page_ok = False
        if existing.get("index_hash") == INDEX_HASH and page_ok:
            _log("already running for the same index; nothing to do")
            return 0
        _log("port already used by a memory-graph service with another index")
        return 2
    try:
        server = GraphHTTPServer((HOST, PORT), Handler)
    except OSError as exc:
        _log("cannot bind %s:%s: %s" % (HOST, PORT, exc))
        return 2
    if WATCH_STDIN:
        # C# keeps our stdin pipe open; when Alife exits, EOF lets us clean up
        # the sidecar even if the plugin's OnDestroy was not called.
        def _watch_stdin():
            try:
                while sys.stdin.readline():
                    pass
            except (OSError, ValueError):
                pass
            _log("parent stdin closed; shutting down")
            try:
                server.shutdown()
            except Exception as e:
                _log("server shutdown failed: %s: %s" % (type(e).__name__, e))

        threading.Thread(target=_watch_stdin, daemon=True).start()
    _log("listening http://%s:%s index_hash=%s no_model=%s" % (HOST, PORT, INDEX_HASH or "-", NO_MODEL))
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.server_close()
        except OSError:
            pass
        _stop_sidecar()
    return 0


if __name__ == "__main__":
    try:
        signal.signal(signal.SIGTERM, lambda *_args: sys.exit(0))
    except (OSError, ValueError):
        pass
    sys.exit(main())
