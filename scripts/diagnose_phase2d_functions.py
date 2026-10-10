"""Privileged function-family representability diagnostic, not a DIC baseline."""
import argparse,json
from pathlib import Path
import numpy as np
from fatigue_dic_rl.correction.law_family_benchmarks import reference_hazard

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rows=[]
    xgrid=np.array([.02,.05,.1,.2,.5,1.,2.,4.,8.,16.,32.])
    for label,family in [('matched','matched'),('parameter','parameter'),('S1','S1'),('S2','S2'),('S3_fixed_R','S3'),('S3_joint_R','S3'),('S4','S4'),('S5','S5')]:
        sets=[(.12,0.,-1.)]
        if label=='S3_fixed_R':sets=[(.12,0.,.3)]
        if label=='S3_joint_R':sets=[(.12,0.,r) for r in (-1.,0.,.3)]
        if label=='S4':sets=[(d,0.,-1.) for d in (.08,.16,.3)]
        if label=='S5':sets=[(.12,x,-1.) for x in (0.,1.,2.)]
        xx=[];yy=[]
        for d,coordinate,R in sets:
            y=reference_hazard(xgrid,np.full(len(xgrid),d),np.c_[np.full(len(xgrid),coordinate),np.zeros(len(xgrid))],R,family=family,A=1.28e-4 if family=='parameter' else 8e-5,m=1.18 if family=='parameter' else 1.,strength={'S2':.5,'S3':.55,'S5':.3}.get(family,3.));xx.extend(xgrid);yy.extend(y)
        x=np.asarray(xx);y=np.asarray(yy);mask=y>0
        coef=np.linalg.lstsq(np.c_[np.ones(mask.sum()),np.log(x[mask])],np.log(y[mask]),rcond=None)[0];A,m=np.exp(coef[0]),coef[1];prediction=A*x**m
        positive_relative=float(np.sqrt(np.mean(((prediction[mask]-y[mask])/y[mask])**2)));zero_false_rate=float(np.max(prediction[~mask])) if (~mask).any() else 0.
        rows.append(dict(label=label,oracle_function_fit_A=float(A),oracle_function_fit_m=float(m),positive_hazard_relative_rmse=positive_relative,zero_hazard_max_false_rate=zero_false_rate,zero_mask_count=int((~mask).sum()),in_declared_DIC_box=bool(1e-5<=A<=5e-4 and .6<=m<=1.6),extra_information='oracle hazard and declared independent state/coordinate/R values; unavailable to DIC selector'))
        np.savez_compressed(out/(label+'.npz'),x=x,true_hazard=y,refit_hazard=prediction,positive_mask=mask)
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,oracle_only=True,note='log-hazard linear least squares without added epsilon; exact-family diagnostic, not a fair observed-prefix calibration score'),indent=2)+'\n')
