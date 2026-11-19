import re
import json
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

class ScratchRAG:
    def __init__(self, embed_model_name: str = "BAAI/bge-small-en-v1.5"):
        self.encoder = SentenceTransformer(embed_model_name)
        self.chunks = []
        self.index = None

    def extract_pages(self, pdf_paths: list[str]) -> list[dict]:
        """Extracts text page-by-page with provenance metadata (>= 30 pages total)."""
        pages = []
        for path in pdf_paths:
            reader = PdfReader(path)
            for page_idx, page in enumerate(reader.pages):
                clean_text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
                if clean_text:
                    pages.append({"doc": path, "page": page_idx + 1, "text": clean_text})
        return pages

    def chunk_pages(self, pages: list[dict], chunk_words: int = 250, overlap_words: int = 50) -> list[dict]:
        """Word-level sliding window chunker preserving document and page citations."""
        assert overlap_words < chunk_words, "Overlap must be smaller than chunk size."
        step = chunk_words - overlap_words
        chunks = []
        chunk_id = 0

        for p in pages:
            words = p["text"].split()
            for start in range(0, len(words), step):
                window = words[start : start + chunk_words]
                if len(window) < 20 and start > 0:
                    continue
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc": p["doc"],
                    "page": p["page"],
                    "text": " ".join(window),
                })
                chunk_id += 1
        return chunks

    def build_index(self, chunks: list[dict]):
        """Encodes chunks and builds an L2-normalized Inner Product (Cosine) FAISS index."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Exact cosine similarity on unit vectors
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieves top-k chunks with similarity scores."""
        q_vec = self.encoder.encode([query], normalize_embeddings=True)
        q_vec = np.asarray(q_vec, dtype=np.float32)
        scores, indices = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                item = dict(self.chunks[idx])
                item["score"] = float(score)
                results.append(item)
        return results

    def generate_answer(self, query: str, retrieved: list[dict], llm_client, model: str) -> str:
        """Grounded generation enforcing strict inline [doc:page] citations."""
        context_block = "\n\n".join(
            f"[Source: {c['doc']}, Page {c['page']}]\n{c['text']}" for c in retrieved
        )
        system_prompt = (
            "Answer the user question using ONLY the provided sources. "
            "Cite every claim as [filename, Page X]. If the answer is not present, say 'Insufficient context.'"
        )
        resp = llm_client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
            ],
        )
        return resp.choices[0].message.content