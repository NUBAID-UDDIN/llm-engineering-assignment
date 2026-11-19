import json
import os
import re
import numpy as np
from rag_pipeline import ScratchRAG


def normalize_for_match(text: str) -> str:
    """Normalizes quotes, dashes, LaTeX/math spacing, and whitespace for fair substring checks."""
    text = text.lower()
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_chunk_config(
    pages: list[dict],
    qa_pairs: list[dict],
    chunk_words: int,
    overlap_words: int,
    k: int = 5,
) -> dict:
    rag = ScratchRAG()
    chunks = rag.chunk_pages(pages, chunk_words=chunk_words, overlap_words=overlap_words)
    rag.build_index(chunks)

    hits_at_k = 0
    reciprocal_ranks = []
    per_query_debug = []

    for item in qa_pairs:
        retrieved = rag.retrieve(item["question"], k=k)
        gold_sub = normalize_for_match(item["gold_passage_substring"])

        found_rank = None
        for rank, cand in enumerate(retrieved, start=1):
            cand_norm = normalize_for_match(cand["text"])
            if gold_sub in cand_norm:
                found_rank = rank
                break

        if found_rank is not None:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        per_query_debug.append({
            "id": item["id"],
            "found_rank": found_rank,
            "top_retrieved_doc_page": [
                f"{c['doc']}:p{c['page']}" for c in retrieved[:3]
            ],
        })

    n = len(qa_pairs)
    return {
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "total_chunks": len(chunks),
        f"Recall@{k}": round(hits_at_k / n, 4),
        f"MRR@{k}": round(float(np.mean(reciprocal_ranks)), 4),
        "details": per_query_debug,
    }


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    pdf_paths = [
        os.path.join(data_dir, "2005.11401v4.pdf"),
        os.path.join(data_dir, "1706.03762v7.pdf"),
    ]
    qa_path = os.path.join(data_dir, "eval_15_qa.json")

    loader = ScratchRAG()
    pages = loader.extract_pages(pdf_paths)
    print(f"Extracted {len(pages)} non-empty pages across {len(pdf_paths)} PDFs.")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    # Config A: Granular chunks (120 words, 25-word overlap)
    res_small = evaluate_chunk_config(pages, qa_pairs, chunk_words=120, overlap_words=25, k=5)

    # Config B: Medium chunks (350 words, 60-word overlap)
    res_medium = evaluate_chunk_config(pages, qa_pairs, chunk_words=350, overlap_words=60, k=5)

    summary = {
        "config_A_120w": {k: v for k, v in res_small.items() if k != "details"},
        "config_B_350w": {k: v for k, v in res_medium.items() if k != "details"},
    }
    print("\n=== SUMMARY COMPARISON ===")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(base_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump({"120w": res_small, "350w": res_medium}, f, indent=2)