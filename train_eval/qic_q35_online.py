"""Qwen3.5 (qic_qwen35, run with PYTHONPATH=py35 / transformers 5) + qic_online options (e.g. --qic_mem), greedy."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qic_qwen35  # noqa: E402,F401  (model class + greedy load)
import qic_online  # noqa: E402  (memory prompt; captures the loader above via qic_greedy)
baseline = qic_online.baseline
if __name__ == "__main__":
    baseline.run(baseline.parse_args())
