"""Independent FEM audit of the limited training condition equivalence."""
import argparse,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from fatigue_dic_rl.training.synthetic_dataset import load_observations
from fatigue_dic_rl.training.uniform_condition import uniform_damage,causal_condition
from fatigue_dic_rl.networks.observation_graph import equivariant_whitening
from fatigue_dic_rl.networks.graph_features import build_graph_frame
from fatigue_dic_rl.networks.temporal_gru import pack_graph_frames
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(62019);rows=[]
    records=json.loads((Path(a.dataset)/'manifest.json').read_text())['records'];records=[r for r in records if r['split']=='train'][:24]
    for i,r in enumerate(records):
        _,predictor,frames=load_observations(r['path']);z=rng.uniform(-.7,.7,2) if i<20 else np.array([.95,.95]);p=predictor.parameter_space.decode(z)
        for frame in frames:
            g=build_graph_frame(frame,z,predictor=predictor,physics=True);m=frame.case.measurement;cov=m.covariance.selected_matrix(m.active_mask);basis=np.column_stack([g.dic.coordinates_mm[:,0],-.3*g.dic.coordinates_mm[:,1]])
            g=replace(g,training_whitened_measured=equivariant_whitening(cov,m.observation.displacement[m.active_mask],0.)[0],training_whitened_affine_basis=equivariant_whitening(cov,basis,0.)[0]);data=pack_graph_frames([g]);equivalent=causal_condition(data,data['previous']);E=next(iter(frame.case.pattern.materials.values())).parameters['young_modulus_n_per_mm2']
            scalar=uniform_damage(p['d0'],p['A'],E,frame.cycle_count);state=predictor.damage_states(z,[frame.cycle_count])[frame.cycle_count];error=float(np.max(abs(state.damage.values-scalar)))
            rows.append({'seed':r['seed'],'cycle':frame.cycle_count,'scalar_vs_FEM_damage_max_error':error,'df_float32_max_error':float(torch.max(abs(data['df']-equivalent['df']))),'ff_float32_max_error':float(torch.max(abs(data['ff']-equivalent['ff']))),'global_float32_max_error':float(torch.max(abs(data['global']-equivalent['global'])))})
            assert error<1e-10 and rows[-1]['df_float32_max_error']<2e-4 and rows[-1]['global_float32_max_error']<2e-5
    (out/'verification.json').write_text(json.dumps({'passed':True,'seed':62019,'rows':rows,'scope':'uniform homogeneous free-contraction x-traction rectangle only; including adaptive large-damage candidates; training conditions only'},indent=2)+'\n');print('passed',len(rows),'FEM comparisons')
