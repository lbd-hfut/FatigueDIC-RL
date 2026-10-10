"""Scientific CLI; all long experiments are outside ordinary pytest."""
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from fatigue_dic_rl.evaluation.correction_scientific import constant_verification,temporal_verification,information_verification,spatial_mesh_verification
from fatigue_dic_rl.training.correction_dataset import source_hash


def task(t):
    function,args=t;return function(*args)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);t=time.perf_counter();metadata=json.loads((root/'training/structural/K4/seed_85000/metadata.json').read_text());coeff=metadata['coefficients'];case=root/'dataset/structural/test/case_84000'
    jobs=[(constant_verification,(out/'constant',)),(temporal_verification,(case,coeff,out/'temporal')),(information_verification,(case,out/'identifiability')),(spatial_mesh_verification,(coeff,out/'spatial_mesh'))]
    with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(task,jobs))
    (out/'execution.json').write_text(json.dumps({'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t,'executed_suites':['constant/zero/semigroup','independent temporal convergence/query invariance','correction_joint_identifiability','spatial_basis/mesh/node-permutation'],'passed':True},indent=2)+'\n');print('scientific primitives passed',flush=True)
