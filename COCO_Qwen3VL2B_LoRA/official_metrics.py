"""Compatibility entrypoint: user-authorized global endpoint micro P/R/F1.
Historical original Rex reproduction lives in legacy_rex_metrics.py.
"""
from micro_metrics import evaluate, load_predictions, save_metrics, self_check

if __name__ == '__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='--native-worker':
        # Already-running generation jobs imported the old Python function.
        # Let their historical scorer finish; dependency jobs rescore the saved
        # predictions using micro_metrics. Never mislabel new values as old ones.
        import runpy
        from pathlib import Path
        runpy.run_path(str(Path(__file__).with_name('legacy_rex_metrics.py')),run_name='__main__')
    else:self_check()
