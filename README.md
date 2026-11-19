# LLM Engineering Mock Assignment

End-to-end engineering submission covering QLoRA fine-tuning, a from-scratch RAG pipeline (no LangChain/LlamaIndex), and quantisation/serving analysis.

## Repository Structure
- `task1_qlora_finetune/`: QLoRA (`r=16, lora_alpha=32`) fine-tuning workflow on `Qwen2.5-1.5B-Instruct`, 50-example held-out evaluation, error analysis, and 7B VRAM math (Full vs. LoRA vs. QLoRA).
- `task2_scratch_rag/`: From-scratch PDF extraction, sliding-window chunking, FAISS (`IndexFlatIP`) retrieval across 34 pages, 15-question evaluation harness (`Recall@5` & `MRR@5`), and hybrid/re-ranker design.
- `task3_serving_quant/`: Precision/quantisation benchmarking (`FP16` vs. `INT4_NF4`), KV-cache & quantisation memory mechanics, and 24 GB GPU concurrency math for 20 users.
- `report/`: Concise `<= 4`-page technical report (`technical_report.pdf`).
