"""Create an isolated diagnostic build; NEVER modify the vendored official engine."""
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

root=Path(sys.argv[1]);source=root/'official_eval/upstream/rex/evaluation/fastevaluate'
target=root/'audits/rex_f1_root_cause_20261102/native_initialized'
if target.exists():raise RuntimeError('Diagnostic directory exists; do not overwrite')
shutil.copytree(source,target,ignore=shutil.ignore_patterns('build','*.so','*.o','__pycache__'))
header=target/'fastevaluate/cocoeval.h';s=header.read_text()
s=s.replace('        double* precision;', '''        vector<double> audit_endpoint_precision, audit_endpoint_recall;
        vector<double> audit_tp, audit_fp, audit_gt;
        double* precision;''',1);header.write_text(s)
path=target/'fastevaluate/cocoeval.cpp';original=path.read_text();s=original
s=s.replace('    precision = new double[T*R*K*A*M];', '''    audit_endpoint_precision.assign(T*K, -1);
    audit_endpoint_recall.assign(T*K, -1);
    audit_tp.assign(T*K, 0); audit_fp.assign(T*K, 0); audit_gt.assign(T*K, 0);
    precision = new double[T*R*K*A*M];''',1)
s=s.replace('            if (npig == 0) continue;', '''            if (npig == 0) {
                if (a == 0 && m == M-1) {
                    for (int t=0; t<T; t++) {
                        double fp_count=0;
                        for (int j=0; j<dtIg[t].size(); j++) if (!dtIg[t][j]) fp_count++;
                        audit_fp[t*K+k]=fp_count;
                    }
                }
                continue;
            }''',1)
s=s.replace('                double q[R]; memset(q, 0, sizeof(q));', '''                if (a == 0 && m == M-1) {
                    audit_endpoint_precision[t*K+k] = nd ? pr[nd-1] : 0;
                    audit_endpoint_recall[t*K+k] = nd ? rc[nd-1] : 0;
                    audit_tp[t*K+k] = nd ? tp[nd-1] : 0;
                    audit_fp[t*K+k] = nd ? fp[nd-1] : 0;
                    audit_gt[t*K+k] = npig;
                }
                double q[R]; memset(q, 0, sizeof(q));''',1)
# Only initialize values that the original precision extraction otherwise reuses.
start=s.index('map<string, vector<double> > COCOEval::accumulate()')
end=s.index('LVISEval::LVISEval(',start);part=s[start:end]
part=part.replace('        ap = 0; ap_sum = 0;', '''        precision50_value = 0; precision95_value = 0;
        ap = 0; ap_sum = 0;''',1)
part=part.replace('        for (int t = 0; t < T; t++) {\n            for (int r = R-1;', '''        for (int t = 0; t < T; t++) {
            precision_value_tmp = 0;
            for (int r = R-1;''',1)
part=part.replace('    result["ap"] = vector_ap;', '''    result["audit_endpoint_precision"] = audit_endpoint_precision;
    result["audit_endpoint_recall"] = audit_endpoint_recall;
    result["audit_tp"] = audit_tp; result["audit_fp"] = audit_fp; result["audit_gt"] = audit_gt;
    vector<double> audit_catids; for (int c : params.catIds) audit_catids.push_back(c);
    result["audit_catids"] = audit_catids;
    result["ap"] = vector_ap;''',1)
s=s[:start]+part+s[end:];path.write_text(s)
if 'precision_value_tmp = 0;' not in part:raise RuntimeError('Initialization diagnostic patch did not apply')
folder=target.parent
(folder/'diagnostic.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='official/cocoeval.cpp',tofile='diagnostic/cocoeval.cpp')))
(folder/'build_manifest.json').write_text(json.dumps({'official_source_sha256':hashlib.sha256(original.encode()).hexdigest(),
 'diagnostic_sha256':hashlib.sha256(s.encode()).hexdigest(),'purpose':'Diagnosis only: reset stale precision values and export existing match counts. Not the formal evaluator.'},indent=2)+'\n')
subprocess.run([sys.executable,'setup.py','build_ext','--inplace'],cwd=target,check=True)
print('DIAGNOSTIC_BUILD_COMPLETE',target,flush=True)
