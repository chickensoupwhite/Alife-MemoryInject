"""Local vector candidate search and optional MiniLM ONNX pairwise reranking."""
import math
import os
import threading
import time
from collections import OrderedDict

import dbside

_CATALOG = {}
_RANK = {"ready": False, "loading": False, "path": "", "error": "", "retry_at": 0}
_LOCK = threading.Lock()
_SCORES = OrderedDict()


def vector_candidates(c, model, query_vector, where, args, accepts, limit=40, minimum=.58):
    """Search every cached, current-version fact vector; SQL owns lifecycle/time filters."""
    import numpy as np
    revision = c.execute("SELECT revision FROM semantic_revision WHERE id=1").fetchone()[0]
    key = (model, int(revision), len(query_vector))
    if _CATALOG.get("connection") is not c or _CATALOG.get("key") != key:
        entries, vectors = [], []
        rows = c.execute(
            "SELECT e.*,v.data AS vector_data FROM entries e JOIN vec v ON v.id=e.id "
            "AND v.version=e.text_hash AND v.model=? WHERE e.retired=0 "
            "AND lower(replace(e.source,char(92),'/')) LIKE 'facts/%'", (model,))
        for row in rows:
            blob = row['vector_data']
            if not blob or len(blob) % 4:
                continue
            vector = np.frombuffer(blob, dtype='<f4')
            if len(vector) != len(query_vector) or not np.isfinite(vector).all():
                continue
            norm = np.linalg.norm(vector)
            if norm == 0:
                continue
            payload = dict(row)
            payload.pop('vector_data')
            entries.append(dbside._row_entry(c, payload))
            vectors.append(vector / norm)
        total = c.execute("SELECT COUNT(*) FROM entries WHERE retired=0 "
                          "AND lower(replace(source,char(92),'/')) LIKE 'facts/%'").fetchone()[0]
        _CATALOG.clear()
        _CATALOG.update(connection=c, key=key, entries=entries,
                        matrix=np.array(vectors, dtype=np.float32), total=total)
    entries = _CATALOG['entries']
    coverage = {'cached': len(entries), 'total': _CATALOG['total'],
                'complete': len(entries) == _CATALOG['total']}
    if not entries:
        return [], coverage
    allowed = {row[0]: row[1] for row in c.execute("SELECT id,text_hash FROM entries WHERE " + ' AND '.join(where), args)}
    q = np.asarray(query_vector, dtype=np.float32)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm == 0:
        return [], coverage
    scores = _CATALOG['matrix'] @ (q / norm)
    # Filter before top-k so excluded people/dates cannot consume the candidate budget.
    candidates = [(float(score), entry) for score, entry in zip(scores, entries)
                  if score >= minimum and allowed.get(entry['id']) == entry['text_hash'] and accepts(entry)]
    candidates.sort(key=lambda pair: (-pair[0], str(pair[1]['id'])))
    return candidates[:limit], coverage


def model_directory(path=''):
    return os.path.abspath(path or os.environ.get('MAO_RERANK_MODEL') or
                           os.path.join(os.path.dirname(__file__), 'models', 'reranker'))


def reranker_status():
    return {key: _RANK[key] for key in ('ready', 'loading', 'path', 'error')}


def start_reranker(path=''):
    path = model_directory(path)
    with _LOCK:
        if _RANK['loading']:
            return
        if _RANK['path'] == path and (_RANK['ready'] or time.monotonic() < _RANK['retry_at']):
            return
        _RANK.update(ready=False, loading=True, path=path, error='')
    threading.Thread(target=_load_reranker, args=(path,), daemon=True,
                     name='retromem-reranker').start()


def _load_reranker(path):
    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(os.path.join(path, 'tokenizer.json'))
        tokenizer.enable_truncation(max_length=512, strategy='longest_first')
        tokenizer.enable_padding(pad_id=1, pad_token='<pad>')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(os.path.join(path, 'onnx', 'model_quint8_avx2.onnx'),
                                      sess_options=options, providers=['CPUExecutionProvider'])
        # Exercise the tokenizer and session before advertising readiness.
        with _LOCK:
            _RANK.update(tokenizer=tokenizer, session=session)
            _predict_pairs([('预热', '本地记忆精排预热。')])
            _SCORES.clear()
            _RANK.update(ready=True, path=path, error='')
    except Exception as exc:
        with _LOCK:
            _RANK.update(ready=False, error=f'{type(exc).__name__}: {exc}',
                         retry_at=time.monotonic() + 30)
    finally:
        with _LOCK:
            _RANK['loading'] = False


def _predict_pairs(pairs):
    import numpy as np
    encoded = _RANK['tokenizer'].encode_batch(pairs)
    values = {'input_ids': np.array([e.ids for e in encoded], dtype=np.int64),
              'attention_mask': np.array([e.attention_mask for e in encoded], dtype=np.int64),
              'token_type_ids': np.array([e.type_ids for e in encoded], dtype=np.int64)}
    feed = {item.name: values[item.name] for item in _RANK['session'].get_inputs()}
    logits = _RANK['session'].run(None, feed)[0].reshape(-1)
    if len(logits) != len(pairs) or not np.isfinite(logits).all():
        raise RuntimeError('invalid reranker output')
    return [1 / (1 + math.exp(-max(-40, min(40, float(value))))) for value in logits]


def rerank(query, entries, path=''):
    path = model_directory(path)
    if not _RANK['ready'] or _RANK['path'] != path:
        start_reranker(path)
        return None
    with _LOCK:
        try:
            result, pending = {}, []
            for entry in entries:
                text = str(entry.get('text') or entry.get('summary') or '')
                key = (query, str(entry['id']), text)
                if key in _SCORES:
                    result[entry['id']] = _SCORES[key]
                    _SCORES.move_to_end(key)
                else:
                    pending.append((key, entry, text))
            for start in range(0, len(pending), 8):
                batch = pending[start:start + 8]
                scores = _predict_pairs([(query, text) for _, _, text in batch])
                for (key, entry, _), score in zip(batch, scores):
                    _SCORES[key] = score
                    result[entry['id']] = score
                while len(_SCORES) > 512:
                    _SCORES.popitem(last=False)
            return result
        except Exception as exc:
            _RANK.update(ready=False, error=f'{type(exc).__name__}: {exc}',
                         retry_at=time.monotonic() + 30)
            return None
