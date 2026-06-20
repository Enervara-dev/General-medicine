"""
Offline ranking-evaluation harness for the episodic memory layer.

Components:
- logger.py  : append-only JSONL writer wired into RetrieverService
- metrics.py : pure ranking metrics (precision@k, recall@k, MRR, nDCG@k)
- cli.py     : interactive labeling + aggregate-metrics CLI

Flow:
1. Flip EPISODIC_EVAL_LOGGING_ENABLED=true in .env.
2. Use the system normally — every retrieval gets logged.
3. Run `python -m episodic.eval.cli label` to mark ground-truth episode IDs
   per logged event.
4. Run `python -m episodic.eval.cli metrics` to see precision@5 / MRR / nDCG
   across all labeled events. Use those numbers to tune EpisodicConfig
   ranker weights.
"""
