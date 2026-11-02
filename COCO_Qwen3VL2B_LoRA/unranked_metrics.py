"""Compatibility import: current evaluation delegates to the original Rex author code.

The prior custom Hungarian/P*R metric is withdrawn; see Git history for that audit.
"""
from official_metrics import evaluate, load_predictions, save_metrics, self_check

if __name__=='__main__':self_check()
