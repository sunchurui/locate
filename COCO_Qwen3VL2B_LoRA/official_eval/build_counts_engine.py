"""Isolated Rex endpoint-count build. Never modify the vendored author source."""
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'official_eval/upstream/rex/evaluation/fastevaluate'
TARGET = ROOT/'official_eval/counts_engine'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Expected one native patch location: {old[:80]}')
    return text.replace(old, new, 1)


def main():
    lock = json.loads((ROOT/'official_eval/sources.lock.json').read_text())
    for relative, sha in lock['files'].items():
        if digest(ROOT/'official_eval/upstream'/relative) != sha:
            raise ValueError(f'Original source changed: {relative}')
    if TARGET.exists():
        raise RuntimeError('Counts engine exists; preserve it and inspect its manifest')
    shutil.copytree(SOURCE, TARGET, ignore=shutil.ignore_patterns('build', '*.so', '*.o', '__pycache__'))
    header = TARGET/'fastevaluate/cocoeval.h'
    header.write_text(once(header.read_text(), '        double* precision;',
        '        vector<double> count_tp, count_fp, count_gt;\n        double* precision;'))
    cpp = TARGET/'fastevaluate/cocoeval.cpp'
    original = cpp.read_text()
    text = once(original, '    precision = new double[T*R*K*A*M];',
        '    count_tp.assign(T*K, 0); count_fp.assign(T*K, 0); count_gt.assign(T*K, 0);\n'
        '    precision = new double[T*R*K*A*M];')
    # Absent-GT categories must still contribute their non-ignored false detections.
    text = once(text, '            if (npig == 0) continue;', '''            if (npig == 0) {
                if (a == 0 && m == M-1) {
                    for (int t=0; t<T; t++) {
                        double false_count=0;
                        for (int j=0; j<dtIg[t].size(); j++) if (!dtIg[t][j]) false_count++;
                        count_fp[t*K+k]=false_count;
                    }
                }
                continue;
            }''')
    # Export existing cumulative TP/FP BEFORE PR interpolation. Matching stays intact.
    text = once(text, '                double q[R]; memset(q, 0, sizeof(q));', '''                if (a == 0 && m == M-1) {
                    count_tp[t*K+k] = nd ? tp[nd-1] : 0;
                    count_fp[t*K+k] = nd ? fp[nd-1] : 0;
                    count_gt[t*K+k] = npig;
                }
                double q[R]; memset(q, 0, sizeof(q));''')
    start = text.index('map<string, vector<double> > COCOEval::accumulate()')
    end = text.index('LVISEval::LVISEval(', start)
    part = text[start:end]
    # Initialize unused legacy extraction to avoid stale/undefined values there.
    part = once(part, '    for (int k = 0; k < K_loop; k++) {\n        ap = 0; ap_sum = 0;',
        '    for (int k = 0; k < K_loop; k++) {\n        precision50_value = 0; precision95_value = 0;\n        ap = 0; ap_sum = 0;')
    part = once(part, '        for (int t = 0; t < T; t++) {\n            for (int r = R-1;',
        '        for (int t = 0; t < T; t++) {\n            precision_value_tmp = 0;\n            for (int r = R-1;')
    part = once(part, '    result["ap"] = vector_ap;', '''    result["count_tp"] = count_tp; result["count_fp"] = count_fp; result["count_gt"] = count_gt;
    vector<double> count_catids; for (int c : params.catIds) count_catids.push_back(c);
    result["count_catids"] = count_catids;
    result["ap"] = vector_ap;''')
    text = text[:start]+part+text[end:]
    cpp.write_text(text)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), text.splitlines(True),
        fromfile='original/cocoeval.cpp', tofile='endpoint-counts/cocoeval.cpp'))
    (TARGET/'counts.patch').write_text(patch)
    subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=TARGET, check=True)
    files = {str(p.relative_to(TARGET)):digest(p) for p in [cpp,header,TARGET/'counts.patch']}
    for p in TARGET.glob('**/*.so'):
        files[str(p.relative_to(TARGET))] = digest(p)
    manifest = {'base_commit':'6508981c1e0c3fbb2dbe7b962a4bb745005f3e2e',
        'builder_sha256':digest(Path(__file__)), 'original_cpp_sha256':digest(SOURCE/'fastevaluate/cocoeval.cpp'),
        'files':files, 'matching_changed':False,
        'changes':'Export endpoint TP/FP/GT including absent-GT FP; initialize unused legacy precision.'}
    (TARGET/'counts_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('COUNTS_ENGINE_BUILT',TARGET)


if __name__ == '__main__':
    main()
