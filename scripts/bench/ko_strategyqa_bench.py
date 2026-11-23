"""Ko-StrategyQA retrieval benchmark through OpenRAG's real HTTP path.

Measures retrieval quality only (no LLM calls). Data is the BEIR-format copy published at
https://huggingface.co/datasets/taeminlee/Ko-StrategyQA (corpus.jsonl, queries.jsonl,
qrels/dev.tsv). It is not vendored here: its card states no license, so download it yourself
into a directory outside the repo, e.g.

    for f in corpus.jsonl queries.jsonl qrels/dev.tsv; do
      curl -fsSL --create-dirs -o "$DATA/$f" \
        "https://huggingface.co/datasets/taeminlee/Ko-StrategyQA/resolve/main/$f"
    done

To measure a real embedding model, start an OpenAI-compatible embeddings server (for example
scripts/bench/st_embed_server.py) and export OPENRAG_TEXT_EMBED_URL / OPENRAG_TEXT_EMBED_MODEL;
the spawned openrag-server inherits them. Without them the deterministic hash embedder is used.

Build the binaries first (cargo build --release -p openrag-server --bins), then run inside
./scripts/test-pg-local.sh (it exports DATABASE_URL / ADMIN_DATABASE_URL /
WORKER_DATABASE_URL for a throwaway PostgreSQL 17):

    ./scripts/test-pg-local.sh python3 -I scripts/bench/ko_strategyqa_bench.py \
        --data "$DATA" --repo . [--limit N] [--profile release|debug] [--out result.json]

Pipeline: seed org/admin -> start openrag-server -> `openrag auth code` + code exchange ->
create bucket -> ingest corpus (title + text) via POST /v2/buckets/{id}/documents ->
run dev queries via POST /v2/buckets/{id}/search -> BEIR-style metrics on documents
(hits deduplicated by document in rank order).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def http(
    method: str,
    url: str,
    body: dict | None = None,
    key: str | None = None,
    timeout: float = 120.0,
):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    if key:
        req.add_header("authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    top10 = ranked[:10]
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(top10) if d in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
    mrr = next((1.0 / (i + 1) for i, d in enumerate(top10) if d in relevant), 0.0)
    return {
        "ndcg@10": dcg / idcg if idcg else 0.0,
        "recall@10": len(set(top10) & relevant) / len(relevant),
        "recall@20": len(set(ranked[:20]) & relevant) / len(relevant),
        "recall@100": len(set(ranked[:100]) & relevant) / len(relevant),
        "mrr@10": mrr,
    }


VARIANTS: dict[str, tuple[str, dict]] = {
    "default": ("server defaults (pipeline config)", {}),
    "old_default": (
        "previous defaults: chunks+thumb+wiki+memory, rrf, rerank",
        {"spaces": ["chunks", "thumb", "wiki", "memory"], "fusion": "rrf", "alpha": 0.3},
    ),
    "chunks_rrf": ("chunks, rrf (BM25 + vector pool)", {"spaces": ["chunks"]}),
    "chunks_vec": ("chunks, convex alpha=0 (vector only)", {"spaces": ["chunks"], "fusion": "convex", "alpha": 0.0}),
    "chunks_bm25": ("chunks, convex alpha=1 (BM25 only)", {"spaces": ["chunks"], "fusion": "convex", "alpha": 1.0}),
    "chunks_cvx05": ("chunks, convex alpha=0.5", {"spaces": ["chunks"], "fusion": "convex", "alpha": 0.5}),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="ingest only the first N corpus rows (+ every relevant row) for a smoke run",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--profile", default="release", choices=["release", "debug"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--variants",
        default="default,chunks_rrf,chunks_vec,chunks_bm25",
        help="comma-separated variant keys to run (see VARIANTS)",
    )
    ap.add_argument(
        "--save-runs",
        type=Path,
        default=None,
        help="write {variant: {query_id: [corpus_id, ...]}} for offline reranking",
    )
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    import psycopg  # noqa: PLC0415  (only needed here)

    corpus = load_jsonl(args.data / "corpus.jsonl")
    queries = {q["_id"]: q["text"] for q in load_jsonl(args.data / "queries.jsonl")}
    qrels: dict[str, set[str]] = {}
    for line in (
        (args.data / "qrels" / "dev.tsv").read_text(encoding="utf-8").splitlines()[1:]
    ):
        qid, cid, score = line.split("\t")
        if int(score) > 0:
            qrels.setdefault(qid, set()).add(cid)
    qrels = {q: r for q, r in qrels.items() if q in queries}
    if args.limit:
        keep = {c for rel in qrels.values() for c in rel}
        corpus = [d for i, d in enumerate(corpus) if i < args.limit or d["_id"] in keep]

    admin_dsn = os.environ["ADMIN_DATABASE_URL"]
    org_id, dept_id, admin_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO iam.orgs (id, slug, name) VALUES (%s, %s, 'Bench')",
            (org_id, f"bench-{str(org_id)[:8]}"),
        )
        conn.execute(
            "INSERT INTO iam.departments (id, org_id, name, arn_path) VALUES (%s, %s, 'Bench', 'bench')",
            (dept_id, org_id),
        )
        conn.execute(
            "INSERT INTO iam.principals (id, org_id, email, display_name, auth_level, is_org_admin) VALUES (%s, %s, %s, 'Bench Admin', 10, true)",
            (admin_id, org_id, f"bench-{str(admin_id)[:8]}@bench.local"),
        )
        conn.execute(
            "INSERT INTO iam.dept_memberships (org_id, dept_id, principal_id, dept_role) VALUES (%s, %s, %s, 'dept_admin')",
            (org_id, dept_id, admin_id),
        )

    bin_dir = args.repo / "target" / args.profile
    cfg_dir = Path(tempfile.mkdtemp(prefix="orag_bench_"))
    op_cfg = cfg_dir / "operator.toml"
    op_cfg.write_text("operator = true\n", encoding="utf-8")
    env = dict(os.environ, OPENRAG_CONFIG=str(op_cfg), OPENRAG_MODE="offline")
    for var in ("OPENRAG_API_KEY", "OPENRAG_API_URL"):
        env.pop(var, None)

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(cfg_dir / "server.log", "wb")
    srv = subprocess.Popen(
        [str(bin_dir / "openrag-server")],
        cwd=args.repo,
        env=dict(env, OPENRAG_BIND=f"127.0.0.1:{port}", RUST_LOG="warn"),
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(200):
            try:
                if http("GET", f"{base}/health", timeout=1)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        else:
            print("server did not become healthy", file=sys.stderr)
            return 1

        out = subprocess.run(
            [str(bin_dir / "openrag"), "auth", "code", "--principal", str(admin_id)],
            cwd=args.repo,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        code = json.loads(out.stdout)["code"]
        st, ex = http("POST", f"{base}/v1/auth/cli/code-exchange", {"code": code})
        assert st == 201, ex
        key = ex["api_key"]
        st, b = http(
            "POST",
            f"{base}/v2/buckets",
            {
                "slug": "ko-strategyqa",
                "name": "Ko-StrategyQA",
                "access_scope": "org",
                "min_auth_level": 1,
                "dept_id": str(dept_id),
            },
            key,
        )
        assert st in (200, 201), b
        bucket = b.get("id") or b.get("bucket_id")

        doc_to_cid: dict[str, str] = {}
        failures: list[str] = []

        def ingest(d: dict) -> None:
            title = d.get("title") or d["_id"]
            st, r = http(
                "POST",
                f"{base}/v2/buckets/{bucket}/documents",
                {
                    "title": title,
                    "text": f"{title}\n{d['text']}",
                    "access_scope": "org",
                    "min_auth_level": 1,
                },
                key,
            )
            if st in (200, 201, 202):
                doc_to_cid[str(r.get("document_id") or r.get("id"))] = d["_id"]
            else:
                failures.append(f"{d['_id']}: {st} {str(r)[:160]}")

        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(ingest, corpus))
        ingest_s = time.monotonic() - t0
        print(
            f"ingested {len(doc_to_cid)}/{len(corpus)} docs in {ingest_s:.1f}s ({len(doc_to_cid) / ingest_s:.1f} docs/s), failures={len(failures)}",
            file=sys.stderr,
        )

        runs: dict[str, dict[str, list[str]]] = {}

        def run(name: str, variant: dict) -> dict:
            per_q: list[dict[str, float]] = []
            lat: list[float] = []
            admitted: list[bool] = []
            max_scores: list[float] = []
            ranked_by_q: dict[str, list[str]] = {}

            def one(qid: str) -> tuple[dict[str, float], float]:
                q = queries[qid]
                t = time.monotonic()
                st, r = http(
                    "POST",
                    f"{base}/v2/buckets/{bucket}/search",
                    {"query": q, "top_k": args.top_k, **variant},
                    key,
                )
                el = time.monotonic() - t
                ranked: list[str] = []
                if st == 200:
                    for h in r.get("results", []):
                        cid = doc_to_cid.get(str(h.get("document_id")))
                        if cid and cid not in ranked:
                            ranked.append(cid)
                ranked_by_q[qid] = ranked
                if st == 200:
                    admitted.append(r.get("admission") == "admit")
                    max_scores.append(float(r.get("max_score") or 0.0))
                return metrics(ranked, qrels[qid]), el

            with cf.ThreadPoolExecutor(args.workers) as pool:
                for m, el in pool.map(one, list(qrels)):
                    per_q.append(m)
                    lat.append(el)
            agg = {k: round(statistics.fmean(m[k] for m in per_q), 4) for k in per_q[0]}
            lat.sort()
            agg["latency_p50_ms"] = round(lat[len(lat) // 2] * 1000, 1)
            agg["latency_p95_ms"] = round(lat[int(len(lat) * 0.95) - 1] * 1000, 1)
            # Every Ko-StrategyQA query has an answer in the corpus, so this is the share of
            # answerable queries the admission threshold lets through (not a false-admit rate).
            agg["admit_rate"] = round(sum(admitted) / max(len(admitted), 1), 4)
            if max_scores:
                agg["max_score_p50"] = round(statistics.median(max_scores), 4)
            runs[name] = ranked_by_q
            return agg

        selected = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in selected if v not in VARIANTS]
        if unknown:
            print(f"unknown variants: {unknown}; known: {list(VARIANTS)}", file=sys.stderr)
            return 2
        plan = [(f"{k}: {VARIANTS[k][0]}", VARIANTS[k][1]) for k in selected]
        report = {
            "dataset": "taeminlee/Ko-StrategyQA (dev)",
            "corpus_docs_ingested": len(doc_to_cid),
            "corpus_docs_total": len(corpus),
            "queries": len(qrels),
            "ingest_seconds": round(ingest_s, 1),
            "ingest_failures": failures[:20],
            "embedder": (
                f"{os.environ['OPENRAG_TEXT_EMBED_MODEL']} via {os.environ['OPENRAG_TEXT_EMBED_URL']} (MRL-truncated to 256d)"
                if os.environ.get("OPENRAG_TEXT_EMBED_URL")
                else "deterministic-hash-256d-v1 (feature hashing, not a semantic model)"
            ),
            "build_profile": args.profile,
            "top_k": args.top_k,
            "results": {name: run(name, body) for name, body in plan},
        }
        if args.save_runs:
            args.save_runs.write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
        return 0
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()
        log.close()


if __name__ == "__main__":
    sys.exit(main())
