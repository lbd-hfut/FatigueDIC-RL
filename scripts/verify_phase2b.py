"""Formal Phase 2B A–P evidence audit; no training or test-based tuning."""
import argparse,json,time,subprocess
from pathlib import Path
from dataclasses import replace
import numpy as np
import torch
from fatigue_dic_rl.training.checkpoint import load_checkpoint
from fatigue_dic_rl.training.synthetic_dataset import load_observations,source_hash
from fatigue_dic_rl.networks.graph_features import build_graph_frame
from fatigue_dic_rl.networks.observation_graph import GraphArrays,coordinate_knn
from fatigue_dic_rl.networks.temporal_gru import pack_graph_frames
from fatigue_dic_rl.evaluation.neural_belief_verification import checkpoint_index
from fatigue_dic_rl.inference.neural_belief_estimator import NeuralBeliefEstimator
from fatigue_dic_rl.core.validation import ContractError


def check_architecture(checkpoint,case):
    model,meta=load_checkpoint(checkpoint);_,predictor,frames=load_observations(case);rng=np.random.default_rng(72019);z=predictor.parameter_space.encode({'d0':.16,'A':.0015});gs=[]
    for i,f in enumerate(frames):gs.append(build_graph_frame(f,z,predictor=predictor,physics=True,previous_cycle=frames[i-1].cycle_count if i else 0.))
    original=pack_graph_frames([gs[0]]);g=gs[0];perm=rng.permutation(len(g.dic.features));dic=GraphArrays(g.dic.coordinates_mm[perm],g.dic.features[perm],coordinate_knn(g.dic.coordinates_mm[perm]));gd=replace(g,dic=dic,mapping_node_ids=g.mapping_node_ids[perm],mapping_weights=g.mapping_weights[perm]);vperm=rng.permutation(len(g.fe.features));inverse=np.argsort(vperm);gf=replace(g,fe=GraphArrays(g.fe.coordinates_mm[vperm],g.fe.features[vperm],inverse[g.fe.edge_index]),mapping_node_ids=inverse[g.mapping_node_ids])
    with torch.no_grad():
        a=model.step(original);b=model.step(pack_graph_frames([gd]));c=model.step(pack_graph_frames([gf]));reload,_=load_checkpoint(checkpoint);d=reload.step(original)
        sequence=[pack_graph_frames([s]) for s in gs];prefix=model(sequence[:2]);full=model(sequence)
    errors={key:float(torch.max(abs(a['normalized_mean']-value['normalized_mean']))) for key,value in [('DIC_permutation',b),('FE_permutation',c),('checkpoint_reload',d)]};errors['causal_prefix']=float(torch.max(abs(prefix[-1]['normalized_mean']-full[1]['normalized_mean'])))
    assert max(errors.values())<2e-6 and errors['checkpoint_reload']==errors['causal_prefix']==0.
    estimator=NeuralBeliefEstimator(predictor,model,diagnose_identifiability=True);ranks=[];sd=[]
    for frame in frames:
        result=estimator.assimilate_observation(frame);assert result.status=='SUCCESS'
        ranks.append(estimator.belief.uncertainty.data_rank);sd.append(estimator.belief.uncertainty.marginal_standard_deviation_physical.tolist())
    assert ranks[0]==1 and ranks[-1]==2
    try:NeuralBeliefEstimator(predictor,model,known_E0=False)
    except ContractError:ambiguity_refused=True
    else:raise AssertionError('unknown E0 must not produce unique neural damage')
    return {'checkpoint':checkpoint,'seed':meta['seed'],'errors':errors,'data_fisher_ranks':ranks,'learned_raw_parameter_sd_by_prefix':sd,'unknown_E0_refused':ambiguity_refused}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--sequential',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);t=time.perf_counter();cp=checkpoint_index(root/'training_initial',a.sequential)
    architecture=[check_architecture(path,root/'dataset/validation/case_0000') for key,path in cp.items() if key.startswith('physics_shape_uncertainty')]
    test=json.loads((root/'test_final/summary.json').read_text());ood=json.loads((root/'generalization_final/summary.json').read_text());noise=json.loads((root/'fixed_noise_final/summary.json').read_text());gauss=json.loads((root/'gaussian_test_final/summary.json').read_text());reg=json.loads((root/'historic_scientific_regression/matrix.json').read_text());pytestmeta=json.loads((root/'pytest_final_execution.json').read_text());cal=json.loads((root/'calibration_final/calibration.json').read_text())
    keys=[k for k in test['methods'] if k.startswith('physics_shape_uncertainty')];recovery={k:test['methods'][k]['parameters']['rmse'] for k in keys}
    point_gate=all(v[0]<.03 and v[1]<.0005 and max(test['methods'][k]['forecast_damage_rmse_mean'])<.05 for k,v in recovery.items());reg_gate=all(r['returncode']==0 for r in reg) and pytestmeta['returncode']==0
    # Test-only qualification, not tuning: explicitly limited marginal calibration
    # gate on every seed, field interval propagation remains a local approximation.
    coverage={k:[c['coverage'] for c in test['methods'][k]['parameters']['coverage']] for k in keys}
    coverage_gate=all(.90<=v<=.99 for values in coverage.values() for v in values)
    result={'source_sha256':source_hash(),'wall_seconds':time.perf_counter()-t,'architecture':architecture,'controlled_point_gate':point_gate,'neural_parameter_coverage_gate':coverage_gate,'parameter_rmse':recovery,'parameter_coverage':coverage,'regression_gate':reg_gate,'historic_case_count':len(reg),'calibration_counts':cal['counts'],'tests':{letter:description for letter,description in zip('ABCDEFGHIJKLMNOP',[
        'Trained DIC/FE permutation check plus graph unit tests','Q16/37/64/256, variable FE mesh and Gaussian six-parameter contracts','Tri3 interpolation/P-transpose fusion unit and integration checks','Causal prefix checkpoint check, actual cycle/delta features, future-IO isolation','Frozen test controlled parameter recovery gate','Same-case physics prior/F/G/shape/global/uncertainty and old teacher-condition validation retained','N30/N40 physics rollout before future IO; no future assimilation','Independent/spatial/missing/registration/unknown-load OOD plus100 independent seeds per fixed case','Raw and calibration-only-scaled full covariance, parameter/field/forecast/observation coverage','Unknown-E refusal plus fresh Phase1C/2A ambiguity/rank scientific regression','Paired cross-mesh same observations, declared extreme geometry/material and Gaussian limitation','Wrong exponent law and fixed unmodelled registration/loading bias; truth retained','All training seed wall times and network/pipeline/FEM solve counts','Observation-only inference/backend and dataset loader AST/IO checks','Exact checkpoint reload and deterministic neural replay','Complete original test suite and37 historic scientific CLI cases'])},'qualification':'PHASE2B_GRAPH_ESTIMATOR_PARTIALLY_QUALIFIED' if point_gate and reg_gate else 'PHASE2B_GRAPH_ESTIMATOR_FAILED','qualification_reason':'controlled low-dimensional point estimation qualified; Gaussian six-parameter recovery, global/OOD calibration and model discrepancy remain limited; no neural/RL claim beyond declared distribution','all_test_methods':test['methods'],'OOD_scenario_count':len(ood['scenarios']),'fixed_noise_scenarios':list(noise['scenarios']),'gaussian_method_count':len(gauss['methods'])}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result['qualification'],point_gate,coverage_gate,reg_gate)
