# -*- coding: utf-8 -*-
"""Mao.MemoryInject —— 向量化服务（常驻子进程）。

协议：stdin 每行一个 JSON 请求，stdout 每行一个 JSON 响应。
  {"id":1,"fn":"embed","texts":["...","..."]}  ->  {"id":1,"ok":true,"vectors":[[...],...]}
  {"id":2,"fn":"ping"}                          ->  {"id":2,"ok":true,"device":"cuda","dim":512}
  {"fn":"__shutdown__"}                         ->  退出
模型：BAAI/bge-small-zh-v1.5（512 维，中文短文本），与官方记忆插件同款。
"""
import sys
import os
import json

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def find_model_dir():
    env = os.environ.get("MAO_EMBED_MODEL")
    if env and os.path.isdir(env):
        return env
    cache = os.environ.get("MODELSCOPE_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "modelscope", "hub")
    cands = [
        os.path.join(cache, "models", "BAAI", "bge-small-zh-v1.5"),
        os.path.join(cache, "models", "BAAI", "bge-small-zh-v1___5"),
    ]
    for c in cands:
        if os.path.isdir(c):
            return c
    raise RuntimeError("未找到 bge-small-zh-v1.5 模型目录：" + cache)


def main():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

    model_dir = find_model_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModel.from_pretrained(model_dir).to(device).eval()

    def embed(texts):
        batch = tok(texts, padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl(**batch)
        vec = out.last_hidden_state[:, 0, :].float()
        vec = F.normalize(vec, dim=-1)
        return vec.cpu().tolist()

    def reply(obj):
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    reply({"id": 0, "fn": "ready", "ok": True, "device": device, "dir": model_dir})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        fn = req.get("fn")
        rid = req.get("id", 0)
        if fn == "__shutdown__":
            break
        try:
            if fn == "ping":
                reply({"id": rid, "ok": True, "device": device, "dim": 512})
            elif fn == "embed":
                texts = req.get("texts") or []
                vecs = embed(texts) if texts else []
                reply({"id": rid, "ok": True, "vectors": vecs})
            else:
                reply({"id": rid, "ok": False, "error": "unknown fn: %s" % fn})
        except Exception as e:
            reply({"id": rid, "ok": False, "error": "%s: %s" % (type(e).__name__, e)})


if __name__ == "__main__":
    main()
