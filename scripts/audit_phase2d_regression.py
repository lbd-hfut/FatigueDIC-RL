"""Exact non-timing comparison of complete frozen Phase2B records."""
import argparse,json
from pathlib import Path
import numpy as np

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();r=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);old=Path('results/phase2b/20261008T114713Z');new=r/'phase2b_regression';rows={}
    for name,before,after in [('uniform','test_final','uniform'),('OOD','generalization_final','generalization'),('fixed_noise','fixed_noise_final','fixed_noise'),('gaussian','gaussian_test_final','gaussian')]:
        original=json.loads((old/before/'records.json').read_text());fresh=json.loads((new/after/'records.json').read_text());lookup={(x['case_id'],x['method']):x for x in fresh};missing=[];status=[];differences=[]
        for x in original:
            key=(x['case_id'],x['method'])
            if key not in lookup:missing.append(key);continue
            y=lookup[key]
            if x['status']!=y['status']:status.append(key)
            for field,v in x.items():
                if field.endswith('_seconds') or v is None or not isinstance(v,(float,int,list)):continue
                try:aa=np.asarray(v,float);bb=np.asarray(y[field],float)
                except (ValueError,TypeError):continue
                if not np.array_equal(aa,bb,equal_nan=True):differences.append(dict(case=key,field=field))
        rows[name]=dict(historic_records=len(original),executed_records=len(fresh),missing=missing,status_changes=status,numeric_differences=differences,passed=not missing and not status and not differences)
    arch=json.loads((new/'architecture.json').read_text());rows['architecture']=dict(checkpoints=len(arch),maximum_error=max(max(x['errors'].values()) for x in arch),passed=len(arch)==5)
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print('passed',all(x['passed'] for x in rows.values()),flush=True)
