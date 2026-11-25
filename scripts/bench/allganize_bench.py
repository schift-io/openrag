"""Allganize RAG-Evaluation-Dataset-KO retrieval benchmark through OpenRAG's real HTTP path.

Measures retrieval only (no LLM calls): 300 Korean questions over public PDFs in five domains
(finance, public, medical, law, commerce), each labeled with the file and page that holds the
answer and whether the answer sits in a paragraph, a table or an image. Data:
https://huggingface.co/datasets/allganize/RAG-Evaluation-Dataset-KO (MIT). The PDFs themselves are
not in the dataset; documents.csv points at the public pages they were taken from. Put them in
<data>/pdfs/<file_name> and the dataset's rag_evaluation_result.csv in <data>/raw/.

The server under test is started from --bin-dir with the caller's environment (DATABASE_URL,
WORKER_DATABASE_URL, ADMIN_DATABASE_URL, OPENRAG_TEXT_EMBED_URL / _MODEL, OPENRAG_PII_KEK_HEX,
OPENRAG_PIPELINE_CONFIG, ...), so the same script measures any build:

    python3 -I scripts/bench/allganize_bench.py --data "$DATA" --repo . \
        --bin-dir target/release [--variants default,hybrid,hyde --hyde-file hyde.json] [--out r.json]

Pipeline: seed org/admin -> start openrag-server -> `openrag auth code` + code exchange ->
create bucket -> upload every PDF (multipart, parsed by the server) -> run the questions through
POST /v2/buckets/{id}/search -> rank of the target file and of the target page.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import re
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
    timeout: float = 600.0,
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


def upload(url: str, key: str, path: Path, timeout: float = 1800.0):
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in (("access_scope", "org"), ("min_auth_level", "1")):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    fname = path.name.replace('"', "")
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n".encode()
        + path.read_bytes()
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(url, data=b"".join(parts), method="POST")
    req.add_header("content-type", f"multipart/form-data; boundary={boundary}")
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


def pages_of(raw: str) -> set[int]:
    return {int(x) for x in re.findall(r"\d+", raw or "")}


def rank_metrics(rank: int | None) -> dict[str, float]:
    return {
        "r@1": float(rank is not None and rank <= 1),
        "r@5": float(rank is not None and rank <= 5),
        "r@10": float(rank is not None and rank <= 10),
        "r@20": float(rank is not None and rank <= 20),
        "mrr@10": (1.0 / rank) if rank is not None and rank <= 10 else 0.0,
    }


VARIANTS: dict[str, tuple[str, dict]] = {
    "default": ("server defaults (pipeline config)", {}),
    "hybrid": ("chunks, rrf (BM25 + vector)", {"spaces": ["chunks"], "fusion": "rrf"}),
    "vector": (
        "chunks, convex alpha=0 (vector only)",
        {"spaces": ["chunks"], "fusion": "convex", "alpha": 0.0},
    ),
    "bm25": (
        "chunks, convex alpha=1 (BM25 only)",
        {"spaces": ["chunks"], "fusion": "convex", "alpha": 1.0},
    ),
    "hyde": ("server defaults + hypothetical_answer from --hyde-file", {}),
    "hybrid_hyde": (
        "chunks, rrf + hypothetical_answer",
        {"spaces": ["chunks"], "fusion": "rrf"},
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--bin-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--variants", default="default")
    ap.add_argument(
        "--hyde-file",
        type=Path,
        default=None,
        help='{"<question index>": "<hypothetical answer>"}',
    )
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-runs", type=Path, default=None)
    ap.add_argument(
        "--dump-pages", type=Path, default=None, help="write page texts as JSONL"
    )
    ap.add_argument(
        "--server-path",
        default=None,
        help="PATH for the server only. The server uses pdftotext and tesseract when they are on PATH, "
        "and the shipped image has neither; pass e.g. /usr/bin:/bin to measure what that image does.",
    )
    args = ap.parse_args()

    import psycopg  # noqa: PLC0415

    rows = list(
        csv.DictReader(
            (args.data / "raw" / "rag_evaluation_result.csv").open(encoding="utf-8")
        )
    )
    pdf_dir = args.data / "pdfs"
    have = {p.name for p in pdf_dir.glob("*.pdf")}
    questions = [
        {
            "idx": i,
            "domain": r["domain"],
            "question": r["question"],
            "file": r["target_file_name"],
            "pages": pages_of(r["target_page_no"]),
            "type": (r.get("context_type") or "").strip() or "unknown",
        }
        for i, r in enumerate(rows)
        if r["target_file_name"] in have
    ]
    hyde = (
        json.loads(args.hyde_file.read_text(encoding="utf-8")) if args.hyde_file else {}
    )

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

    cfg_dir = Path(tempfile.mkdtemp(prefix="orag_allganize_"))
    op_cfg = cfg_dir / "operator.toml"
    op_cfg.write_text("operator = true\n", encoding="utf-8")
    env = dict(os.environ, OPENRAG_CONFIG=str(op_cfg))
    for var in ("OPENRAG_API_KEY", "OPENRAG_API_URL", "OPENRAG_DOCS_DIRS"):
        env.pop(var, None)
    env["OPENRAG_DOCS_DIRS"] = "[]"
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = open(cfg_dir / "server.log", "wb")
    srv_env = dict(env, OPENRAG_BIND=f"127.0.0.1:{port}", RUST_LOG="warn")
    if args.server_path is not None:
        srv_env["PATH"] = args.server_path
    srv = subprocess.Popen(
        [str(args.bin_dir / "openrag-server")],
        cwd=args.repo,
        env=srv_env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(600):
            try:
                if http("GET", f"{base}/health", timeout=1)[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        else:
            print(
                f"server did not become healthy; log: {cfg_dir / 'server.log'}",
                file=sys.stderr,
            )
            return 1
        out = subprocess.run(
            [
                str(args.bin_dir / "openrag"),
                "auth",
                "code",
                "--principal",
                str(admin_id),
            ],
            cwd=args.repo,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        st, ex = http(
            "POST",
            f"{base}/v1/auth/cli/code-exchange",
            {"code": json.loads(out.stdout)["code"]},
        )
        assert st == 201, ex
        key = ex["api_key"]
        st, b = http(
            "POST",
            f"{base}/v2/buckets",
            {
                "slug": "allganize",
                "name": "Allganize",
                "access_scope": "org",
                "min_auth_level": 1,
                "dept_id": str(dept_id),
            },
            key,
        )
        assert st in (200, 201), b
        bucket = b.get("id") or b.get("bucket_id")

        doc_to_file: dict[str, str] = {}
        failures: list[str] = []
        # Every saved PDF goes in, including the dataset's documents that no question targets:
        # they are part of the corpus the questions have to be found in.
        files = sorted(have)

        def ingest(name: str) -> None:
            st, r = upload(f"{base}/v2/buckets/{bucket}/documents", key, pdf_dir / name)
            if st in (200, 201, 202):
                doc_to_file[str(r.get("document_id") or r.get("id"))] = name
            else:
                failures.append(f"{name}: {st} {str(r)[:200]}")

        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(ingest, files))
        ingest_s = time.monotonic() - t0
        print(
            f"ingested {len(doc_to_file)}/{len(files)} files in {ingest_s:.1f}s, failures={len(failures)}",
            file=sys.stderr,
        )

        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            page_no = {
                str(pid): int(no)
                for pid, no in conn.execute(
                    "SELECT id, page_no FROM corpus.pages WHERE org_id = %s", (org_id,)
                ).fetchall()
            }
            chunk_stats = conn.execute(
                "SELECT count(*), avg(length(redacted_text))::int FROM corpus.chunks WHERE org_id = %s",
                (org_id,),
            ).fetchone()
            pages_total = conn.execute(
                "SELECT count(*) FROM corpus.pages WHERE org_id = %s", (org_id,)
            ).fetchone()[0]
            if args.dump_pages:
                # Page text as the parser produced it, for checking extraction and chunking offline.
                with args.dump_pages.open("w", encoding="utf-8") as fh:
                    for doc_id, no, text in conn.execute(
                        "SELECT document_id, page_no, raw_markdown FROM corpus.pages WHERE org_id = %s ORDER BY document_id, page_no",
                        (org_id,),
                    ):
                        fh.write(
                            json.dumps(
                                {
                                    "file": doc_to_file.get(str(doc_id)),
                                    "page": no,
                                    "text": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        runs: dict[str, dict[str, dict]] = {}

        def run(name: str, variant: dict, use_hyde: bool) -> dict:
            per_q: list[dict] = []
            lat: list[float] = []

            def one(q: dict) -> tuple[dict, float]:
                body = {"query": q["question"], "top_k": args.top_k, **variant}
                if use_hyde and hyde.get(str(q["idx"])):
                    body["hypothetical_answer"] = hyde[str(q["idx"])]
                t = time.monotonic()
                st, r = http("POST", f"{base}/v2/buckets/{bucket}/search", body, key)
                el = time.monotonic() - t
                docs: list[str] = []
                pages: list[tuple[str, int]] = []
                if st == 200:
                    for h in r.get("results", []):
                        f = doc_to_file.get(str(h.get("document_id")))
                        if not f:
                            continue
                        if f not in docs:
                            docs.append(f)
                        p = page_no.get(str(h.get("page_id")))
                        if p is not None and (f, p) not in pages:
                            pages.append((f, p))
                doc_rank = next(
                    (i + 1 for i, f in enumerate(docs) if f == q["file"]), None
                )
                page_rank = next(
                    (
                        i + 1
                        for i, (f, p) in enumerate(pages)
                        if f == q["file"] and p in q["pages"]
                    ),
                    None,
                )
                return {
                    "idx": q["idx"],
                    "domain": q["domain"],
                    "type": q["type"],
                    "file": q["file"],
                    "doc": rank_metrics(doc_rank),
                    "page": rank_metrics(page_rank),
                    "doc_rank": doc_rank,
                    "page_rank": page_rank,
                    "status": st,
                }, el

            with cf.ThreadPoolExecutor(args.workers) as pool:
                for m, el in pool.map(one, questions):
                    per_q.append(m)
                    lat.append(el)

            def agg(items: list[dict]) -> dict:
                if not items:
                    return {}
                out = {"n": len(items)}
                for level in ("doc", "page"):
                    for k in items[0][level]:
                        out[f"{level}_{k}"] = round(
                            statistics.fmean(m[level][k] for m in items), 4
                        )
                return out

            lat.sort()
            # Questions on a file that failed to ingest (a scan without OCR) can't be found by any
            # variant; "ingested" leaves them out so variants are compared on what was searchable.
            ingested_files = set(doc_to_file.values())
            searchable = [m for m in per_q if m["file"] in ingested_files]
            res = {
                "all": agg(per_q),
                "ingested": agg(searchable),
                # by_type / by_domain: ingested questions only, as above.
                "by_type": {
                    t: agg([m for m in searchable if m["type"] == t])
                    for t in sorted({m["type"] for m in searchable})
                },
                "by_domain": {
                    d: agg([m for m in searchable if m["domain"] == d])
                    for d in sorted({m["domain"] for m in searchable})
                },
                "latency_p50_ms": round(lat[len(lat) // 2] * 1000, 1),
                "errors": sum(1 for m in per_q if m["status"] != 200),
            }
            runs[name] = {
                str(m["idx"]): {"doc_rank": m["doc_rank"], "page_rank": m["page_rank"]}
                for m in per_q
            }
            return res

        selected = [v.strip() for v in args.variants.split(",") if v.strip()]
        unknown = [v for v in selected if v not in VARIANTS]
        if unknown:
            print(
                f"unknown variants: {unknown}; known: {list(VARIANTS)}", file=sys.stderr
            )
            return 2
        if any("hyde" in v for v in selected) and not hyde:
            print("hyde variants need --hyde-file", file=sys.stderr)
            return 2
        report = {
            "dataset": "allganize/RAG-Evaluation-Dataset-KO (test)",
            "questions": len(questions),
            "questions_total": len(rows),
            "files_ingested": len(doc_to_file),
            "files_total": len(files),
            "pages": pages_total,
            "chunks": chunk_stats[0],
            "chunk_avg_chars": chunk_stats[1],
            "ingest_seconds": round(ingest_s, 1),
            "ingest_failures": failures[:20],
            "embedder": f"{os.environ.get('OPENRAG_TEXT_EMBED_MODEL')} via {os.environ.get('OPENRAG_TEXT_EMBED_URL')}",
            "bin_dir": str(args.bin_dir),
            "top_k": args.top_k,
            "results": {
                f"{k}: {VARIANTS[k][0]}": run(k, VARIANTS[k][1], "hyde" in k)
                for k in selected
            },
        }
        if args.save_runs:
            args.save_runs.write_text(
                json.dumps(runs, ensure_ascii=False), encoding="utf-8"
            )
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
