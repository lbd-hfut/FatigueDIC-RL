from dataclasses import replace
import numpy as np
from fatigue_dic_rl.evaluation.sequential_fixtures import uniform_setup,generate_truth,observation_series,fatigue_loading
from fatigue_dic_rl.inference.belief_state import ObservationFrame
from fatigue_dic_rl.networks.graph_features import build_graph_frame


def test_known_healthy_node_response_and_dic_prediction_agree():
    c={'geometry':{'nx':2,'ny':1,'seed':19},'observation':{'count':16}};model,space,pattern,p,x=uniform_setup(c,fatigue=True);truth=generate_truth(pattern,model,{'d0':.12,'A':.0015},[0.],x,load_history=fatigue_loading(pattern));f=observation_series(pattern,truth,x,sigma_mm=5e-5)[0];f=ObservationFrame('healthy',replace(f.case,known_healthy=True));z=space.encode({'d0':.3,'A':.002});g=build_graph_frame(f,z,predictor=p,physics=True)
    projected=np.einsum('qi,qij->qj',g.mapping_weights,g.fe.features[:,-2:][g.mapping_node_ids]);np.testing.assert_allclose(projected,g.dic.features[:,4:6],atol=1e-12)
