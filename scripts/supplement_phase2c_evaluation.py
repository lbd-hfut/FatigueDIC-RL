"""Expanded fair parameter calibration and declared nuisance/oracle references."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from evaluate_phase2c import evaluate_task,summarize
from fatigue_dic_rl.training.correction_dataset import source_hash

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);ap.add_argument('--suite',choices=['fair','reference','nuisance','spatial'],required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);cases=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];cases=[r for r in cases if r['split'] in ('test','ood')];jobs=[];t=time.perf_counter()
 for case in cases:
  if a.suite=='fair':methods=['B1_refit_A','B2_refit_Am']
  elif a.suite=='reference':methods=['correct_physics_oracle_reference']
  elif a.suite=='spatial':
   if case['scenario']!='spatial_discrepancy':continue
   methods=['ablation_spatial_K2']
  else:
   if case['scenario'] not in ('registration_bias','unknown_loading'):continue
   methods=['B2_refit_Am','B5_multistep_K2']
  for method in methods:
   options={}
   if a.suite=='nuisance':options={'shared_registration':True} if case['scenario']=='registration_bias' else {'load_unknown':True}
   jobs.append((case['path'],method,str(out/'raw'/case['case_id']/method),{'extra_options':options}))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(evaluate_task,jobs))
 (out/'records.json').write_text(json.dumps(rows,indent=2)+'\n');(out/'summary.json').write_text(json.dumps({'scenarios':summarize(rows),'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t,'failed':sum(r['status']=='FAILED' for r in rows),'suite':a.suite},indent=2)+'\n');print(a.suite,len(rows),'complete',flush=True)
