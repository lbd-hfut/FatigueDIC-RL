"""Tri3 connectivity graph, with node/element/quadrature supports kept explicit."""
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from .observation_graph import GraphArrays


def fe_connectivity_edges(mesh):
    if any(str(t.value if hasattr(t,'value') else t)!='tri3' for t in mesh.element_types):
        raise ContractError('neural FE graph currently supports Tri3')
    edges=set()
    for cell in mesh.connectivity:
        for i in cell:
            for j in cell:
                if i!=j:edges.add((int(i),int(j)))
    return np.asarray(sorted(edges),dtype=int).T


def fe_node_features(pattern, predicted_displacement_mm=None):
    mesh=pattern.mesh;v=mesh.n_nodes;bc=np.zeros((v,2));material=np.zeros((v,3));weight=np.zeros(v)
    for b in pattern.displacement_bcs:bc[b.node_ids,b.components]=1.
    for cell,region in zip(mesh.connectivity,mesh.material_region_ids):
        p=pattern.materials[int(region)].parameters
        material[cell]+=np.array([p['young_modulus_n_per_mm2']/2000.,p['poisson_ratio'],float(region)])
        weight[cell]+=1
    if np.any(weight==0):raise ContractError('isolated FE node not supported')
    material/=weight[:,None]
    u=np.zeros((v,2)) if predicted_displacement_mm is None else np.asarray(predicted_displacement_mm)
    if u.shape!=(v,2):raise ContractError('node displacement must be explicit [V,2], never element/quadrature')
    return GraphArrays(mesh.nodes,np.column_stack([mesh.nodes/1.,bc,material,u/.01]),fe_connectivity_edges(mesh))
