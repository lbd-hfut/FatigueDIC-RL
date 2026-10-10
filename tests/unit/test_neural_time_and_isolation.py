import ast
from pathlib import Path
import numpy as np
import pytest
from fatigue_dic_rl.networks.graph_features import build_graph_frame
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup,generate_truth,observation_series,fatigue_loading


def test_physical_cycle_feature_not_observation_index():
    c={'geometry':{'nx':2,'ny':1,'seed':19},'observation':{'count':16}};model,space,pattern,p,x=uniform_setup(c,fatigue=True)
    truth=generate_truth(pattern,model,{'d0':.12,'A':.0015},[5.,20.],x,load_history=fatigue_loading(pattern));frames=observation_series(pattern,truth,x,sigma_mm=5e-5);z=space.encode({'d0':.16,'A':.0015})
    a=build_graph_frame(frames[0],z,previous_cycle=0.);b=build_graph_frame(frames[1],z,previous_cycle=5.);np.testing.assert_allclose(a.global_features[18:20],[.125,.125]);np.testing.assert_allclose(b.global_features[18:20],[.5,.375])


def test_inference_has_no_truth_or_training_imports():
    path=Path('src/fatigue_dic_rl/inference/neural_belief_estimator.py');tree=ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node,ast.ImportFrom):assert not any(x in (node.module or '').split('.') for x in ('evaluation','training','rl'))
    assert 'load_labels' not in path.read_text() and 'synthetic_labels' not in path.read_text()


def test_no_torch_dependency_on_numerical_path():
    import subprocess,sys
    code="import sys;from fatigue_dic_rl.inference import PhysicsBasedBeliefEstimator; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable,'-c',code],check=True)
