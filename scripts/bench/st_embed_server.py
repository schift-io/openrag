"""Minimal OpenAI-compatible /v1/embeddings server over sentence-transformers.

Lets openrag-server use a local Hugging Face embedding model through HttpTextEmbedder
(OPENRAG_TEXT_EMBED_URL). Requests arriving together are micro-batched into one encode call.
Input text is embedded as sent: OpenRAG adds the model's task prefixes itself.

Needs sentence-transformers>=6.1 / transformers>=5.18 for google/embeddinggemma-2; install them
into an isolated venv rather than the system interpreter, e.g.

    uv venv .venv-embed && VIRTUAL_ENV=.venv-embed uv pip install "sentence-transformers>=6.1" torch
    .venv-embed/bin/python scripts/bench/st_embed_server.py --model google/embeddinggemma-2 --port 8090

then start openrag-server with

    OPENRAG_TEXT_EMBED_URL=http://127.0.0.1:8090/v1/embeddings
    OPENRAG_TEXT_EMBED_MODEL=google/embeddinggemma-2
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Batcher:
    def __init__(self, model, max_batch: int, wait_s: float):
        self.model = model
        self.max_batch = max_batch
        self.wait_s = wait_s
        self.q: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def embed(self, texts: list[str]) -> list[list[float]]:
        done = threading.Event()
        slot: dict = {}
        self.q.put((texts, slot, done))
        done.wait()
        if "error" in slot:
            raise RuntimeError(slot["error"])
        return slot["vectors"]

    def _run(self) -> None:
        while True:
            jobs = [self.q.get()]
            n = len(jobs[0][0])
            while n < self.max_batch:
                try:
                    job = self.q.get(timeout=self.wait_s)
                except queue.Empty:
                    break
                jobs.append(job)
                n += len(job[0])
            flat = [t for texts, _, _ in jobs for t in texts]
            try:
                vecs = self.model.encode(
                    flat,
                    batch_size=self.max_batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).tolist()
                i = 0
                for texts, slot, _ in jobs:
                    slot["vectors"] = vecs[i : i + len(texts)]
                    i += len(texts)
            except Exception as e:  # noqa: BLE001 - reported back to each caller
                for _, slot, _ in jobs:
                    slot["error"] = str(e)
            for _, _, done in jobs:
                done.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--wait-ms", type=float, default=5.0)
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="bfloat16 matched float32 quality for embeddinggemma-2 on Ko-StrategyQA "
        "(nDCG@10 0.8006 vs 0.8012 at 256d) and indexed ~1.65x faster on Apple MPS",
    )
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs: dict = {"model_kwargs": {"dtype": getattr(torch, args.dtype)}}
    if "embeddinggemma-2" in args.model:
        # Text-only: skip the vision and audio encoders (model card, "Selective Encoder Loading").
        kwargs["config_kwargs"] = {"vision_config": None, "audio_config": None}
    model = SentenceTransformer(args.model, device=args.device, **kwargs)
    batcher = Batcher(model, args.max_batch, args.wait_ms / 1000)
    model_id = args.model

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # quiet
            pass

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/health", "/v1/models"):
                self._send(
                    200,
                    {"object": "list", "data": [{"id": model_id, "object": "model"}]},
                )
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path.rstrip("/") not in ("/v1/embeddings", "/embeddings"):
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                req = json.loads(
                    self.rfile.read(int(self.headers.get("content-length", "0")))
                )
                inp = req.get("input")
                texts = [inp] if isinstance(inp, str) else list(inp or [])
                if not texts or not all(isinstance(t, str) for t in texts):
                    raise ValueError("'input' must be a string or a list of strings")
                vecs = batcher.embed(texts)
                dim = req.get("dimensions")
                if isinstance(dim, int) and 0 < dim < len(vecs[0]):
                    arr = np.asarray(vecs, dtype=np.float32)[:, :dim]
                    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
                    vecs = arr.tolist()
            except ValueError as e:
                self._send(400, {"error": {"message": str(e)}})
                return
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": {"message": str(e)}})
                return
            self._send(
                200,
                {
                    "object": "list",
                    "model": model_id,
                    "data": [
                        {"object": "embedding", "index": i, "embedding": v}
                        for i, v in enumerate(vecs)
                    ],
                },
            )

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"serving {model_id} on http://{args.host}:{args.port}/v1/embeddings ({model.device})",
        file=sys.stderr,
        flush=True,
    )
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
