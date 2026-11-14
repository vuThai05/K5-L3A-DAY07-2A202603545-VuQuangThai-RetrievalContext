"""Compare Dense (Day 6 pgvector), BM25, and Hybrid (RRF) on one DB strategy.

Run from the Day 7 project root. Uses the existing Day 6 rag_chunks table;
never writes to or re-embeds the document vectors.
"""
from __future__ import annotations
import argparse, json, math, os, re, statistics, time, unicodedata
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()  # fallback to project-root .env


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    # Simple whitespace/punctuation tokenizer; BM25 remains lexical baseline.
    return re.findall(r"[\w]+", unicodedata.normalize("NFC", text).casefold(), flags=re.UNICODE)


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    body = normalize(chunk["text"])
    for item in gold:
        if (chunk["source_id"] == item["source_id"] and
            int(chunk["page"]) == int(item["page"]) and
            normalize(item["evidence"]) in body):
            return True
    return False


def pack_context(results: list[dict], budget: int) -> tuple[str, list[str]]:
    blocks, ids = [], []
    for c in results:
        block = f'[{c["id"]} | {c["source"]} | page {c["page"]}]\n{c["text"]}'
        candidate = "\n\n".join(blocks + [block])
        if len(candidate) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids


def load_db_chunks(database_url: str, strategy: str, dimensions: int) -> list[dict]:
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise RuntimeError("Cài psycopg[binary] và pgvector theo requirements.txt") from exc
    # Validate vector dimension before using the existing index.
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                WHERE a.attrelid = 'rag_chunks'::regclass
                  AND a.attname = 'embedding' AND NOT a.attisdropped
            """)
            row = cur.fetchone()
            actual = row[0] if row else None
            if actual != f"vector({dimensions})":
                raise RuntimeError(f"Dimension DB={actual}; cấu hình yêu cầu vector({dimensions}). Dừng để tránh dùng sai model.")
            cur.execute("""
                SELECT chunk_id, document_id, source, page, content, metadata
                FROM rag_chunks WHERE strategy = %s
                ORDER BY chunk_id
            """, (strategy,))
            rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"Không có chunk trong rag_chunks với strategy={strategy!r}.")
    return [{"id": r[0], "source_id": r[1], "source": r[2], "page": r[3],
             "text": r[4], "metadata": r[5] if isinstance(r[5], dict) else {}}
            for r in rows]


class DenseSearch:
    def __init__(self, database_url: str, strategy: str, dimensions: int):
        import psycopg
        from pgvector.psycopg import register_vector
        self.conn = psycopg.connect(database_url)
        register_vector(self.conn)
        self.strategy = strategy
        self.dimensions = dimensions

    def search(self, query_vector: list[float], top_k: int) -> list[dict]:
        from pgvector import Vector
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector dimension {len(query_vector)} != {self.dimensions}")
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, document_id, source, page, content, metadata,
                       1 - (embedding <=> %s) AS score
                FROM rag_chunks
                WHERE strategy = %s
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (Vector(query_vector), self.strategy, Vector(query_vector), top_k))
            rows = cur.fetchall()
        return [{"id": r[0], "source_id": r[1], "source": r[2], "page": r[3],
                 "text": r[4], "metadata": r[5] if isinstance(r[5], dict) else {},
                 "score": float(r[6]), "method": "dense"} for r in rows]

    def close(self):
        self.conn.close()


class BM25Search:
    def __init__(self, chunks: list[dict]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("Cài rank-bm25: pip install rank-bm25") from exc
        self.chunks = chunks
        self.model = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, question: str, top_k: int) -> list[dict]:
        scores = self.model.get_scores(tokenize(question))
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.chunks[i]["id"]))
        results = []
        for i in order:
            score = float(scores[i])
            if score <= 0:
                continue
            results.append({**self.chunks[i], "score": score, "method": "bm25"})
            if len(results) == top_k:
                break
        return results


def hybrid_rrf(dense: list[dict], bm25: list[dict], top_k: int, k: int = 60) -> list[dict]:
    by_id, fused = {}, {}
    for rank, item in enumerate(dense, 1):
        by_id[item["id"]] = item
        fused[item["id"]] = fused.get(item["id"], 0.0) + 1.0 / (k + rank)
    for rank, item in enumerate(bm25, 1):
        by_id[item["id"]] = item
        fused[item["id"]] = fused.get(item["id"], 0.0) + 1.0 / (k + rank)
    ranked = sorted(fused, key=lambda cid: (-fused[cid], cid))[:top_k]
    return [{**by_id[cid], "score": fused[cid], "method": "hybrid_rrf"} for cid in ranked]


def evaluate(method: str, questions: list[dict], search_fn, top_k: int, budget: int) -> tuple[list[dict], dict]:
    traces = []
    for q in questions:
        started = time.perf_counter()
        results = search_fn(q["question"], top_k)
        elapsed = (time.perf_counter() - started) * 1000
        marked = [{**c, "hit": is_hit(c, q["gold"])} for c in results]
        context, context_ids = pack_context(results, budget)
        traces.append({"query_id": q["id"], "split": q["split"], "question": q["question"],
                       "method": method, "top_k": top_k, "budget_chars": budget,
                       "gold": q["gold"], "results": marked, "context": context,
                       "context_ids": context_ids, "context_chars": len(context),
                       "elapsed_ms": round(elapsed, 3), "hit1": bool(marked and marked[0]["hit"]),
                       "hit3": any(x["hit"] for x in marked),
                       "context_hit": any(x["hit"] and x["id"] in context_ids for x in marked)})
    n = len(traces)
    summary = {"method": method, "n": n, "hit1": sum(x["hit1"] for x in traces),
               "hit1_rate": round(sum(x["hit1"] for x in traces) / n, 4) if n else 0,
               "hit3": sum(x["hit3"] for x in traces),
               "hit3_rate": round(sum(x["hit3"] for x in traces) / n, 4) if n else 0,
               "context_hit": sum(x["context_hit"] for x in traces),
               "context_hit_rate": round(sum(x["context_hit"] for x in traces) / n, 4) if n else 0,
               "avg_context_chars": round(statistics.mean(x["context_chars"] for x in traces), 2) if n else 0,
               "median_ms": round(statistics.median(x["elapsed_ms"] for x in traces), 3) if n else 0}
    return traces, summary


def main():
    parser = argparse.ArgumentParser(description="Day 7: Dense vs BM25 vs Hybrid RRF")
    parser.add_argument("--questions", type=Path, default=Path("starter/data/converted/questions.json"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--strategy", choices=["A", "B", "C"], default="B",
                        help="Day 6 chunking namespace; default B (legal-structure chunks)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Candidates per retriever before RRF; final result remains top-k")
    parser.add_argument("--context-budget", type=int, default=800)
    parser.add_argument("--output", type=Path, default=Path("starter/results/retrieval"))
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k:
        parser.error("Yêu cầu top-k >= 1 và candidate-k >= top-k")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    load_dotenv()  # fallback to project-root .env
    db_url = os.getenv("DATABASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    if not db_url:
        raise RuntimeError("Thiếu DATABASE_URL trong .env của Day 7 (copy từ Day 6).")
    if provider != "openai" or not api_key:
        raise RuntimeError("Script hiện dùng OpenAI query embedding; kiểm tra EMBEDDING_PROVIDER=openai và OPENAI_API_KEY.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Cài openai>=1,<3 theo requirements.") from exc
    if not args.questions.is_file():
        raise FileNotFoundError(args.questions)
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    questions = [q for q in questions if q.get("split") == args.split]
    if not questions:
        raise ValueError(f"Không có câu hỏi split={args.split}")
    if any(not q.get("gold") for q in questions):
        raise ValueError("questions.json phải chỉ chứa answerable questions có gold. Unanswerable đánh giá riêng ở Agent.")

    chunks = load_db_chunks(db_url, args.strategy, dimensions)
    bm25 = BM25Search(chunks)
    dense = DenseSearch(db_url, args.strategy, dimensions)
    client = OpenAI(api_key=api_key, timeout=60, max_retries=2)
    query_vectors = {}
    def embed_query(question: str) -> list[float]:
        if question not in query_vectors:
            response = client.embeddings.create(model=model, input=[question], dimensions=dimensions)
            query_vectors[question] = response.data[0].embedding
        return query_vectors[question]

    def dense_fn(question, k):
        return dense.search(embed_query(question), k)
    def bm25_fn(question, k):
        return bm25.search(question, k)
    def hybrid_fn(question, k):
        d = dense_fn(question, args.candidate_k)
        b = bm25_fn(question, args.candidate_k)
        return hybrid_rrf(d, b, k)

    all_summaries = []
    try:
        for method, fn in [("dense", dense_fn), ("bm25", bm25_fn), ("hybrid_rrf", hybrid_fn)]:
            traces, summary = evaluate(method, questions, fn, args.top_k, args.context_budget)
            summary.update({"split": args.split, "strategy": args.strategy,
                            "chunks_in_namespace": len(chunks), "embedding_model": model,
                            "embedding_dimensions": dimensions, "candidate_k": args.candidate_k if method == "hybrid_rrf" else None,
                            "top_k": args.top_k, "context_budget_chars": args.context_budget})
            out = args.output / args.split / method
            out.mkdir(parents=True, exist_ok=True)
            (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            all_summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False))
    finally:
        dense.close()
    comparison = {"split": args.split, "strategy": args.strategy, "top_k": args.top_k,
                  "candidate_k": args.candidate_k, "context_budget_chars": args.context_budget,
                  "methods": all_summaries}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / args.split / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nComparison saved: {args.output / args.split / 'comparison.json'}")

if __name__ == "__main__":
    main()
