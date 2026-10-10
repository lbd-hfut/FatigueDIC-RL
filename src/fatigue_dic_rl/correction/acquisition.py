"""Observation-only acquisition loader with explicit public mesh and planned loads.

This module has no truth/training/evaluation imports. Prefix selection precedes
all observation IO; unseen displacement files are never opened by calibration.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import numpy as np
from fatigue_dic_rl.core import LoadingState
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.geometry import MeshData, BoundaryTag
from fatigue_dic_rl.materials import IsotropicPlaneStressMM
from fatigue_dic_rl.mechanics import MaterialAssignment, DisplacementBC, TractionBC
from fatigue_dic_rl.mechanics.tri3_contracts import Tri3SolverRequest
from fatigue_dic_rl.fatigue import FatigueLoadCycle, LoadingMode
from fatigue_dic_rl.inverse.parameterizations import UniformDamage, Parameter, ParameterSpace
from fatigue_dic_rl.inverse.forward_adapter import StaticLoadCase
from fatigue_dic_rl.observations.noise import DICMeasurement
from fatigue_dic_rl.inference.belief_state import ObservationFrame, ObservationBatch, ObservationDesign
from .hybrid_damage_law import LawSpecification
from .physics_predictor import CorrectionPredictor

PUBLIC_KEYS = {'schema', 'case_id', 'split', 'seed', 'mesh', 'material', 'load_scales', 'R',
    'cycles', 'assimilation_cutoff', 'future_load_factor', 'step', 'nominal', 'sigma_mm',
    'noise_model', 'source_sha256', 'config_sha256', 'units', 'shared_registration'}


@dataclass(frozen=True)
class CorrectionAcquisition:
    metadata: dict
    predictors: tuple
    batches: tuple

    def future_designs(self, *, cycles=None):
        times = cycles or tuple(n for n in self.metadata['cycles'] if n > self.metadata['assimilation_cutoff'])
        result = []
        for i, (p, b) in enumerate(zip(self.predictors, self.batches)):
            if self.metadata['future_load_factor'] != p.load_history[1].load_max:
                raise ContractError('public future load plan was changed after predictor construction')
            f = b.frames[-1]
            designs = []
            for n in times:
                if n <= self.metadata['assimilation_cutoff']:
                    raise ContractError('future designs must follow the fixed cutoff')
                pattern = replace(p.pattern, traction_bcs=tuple(replace(bc, values=bc.values*self.metadata['future_load_factor']) for bc in p.pattern.traction_bcs))
                designs.append(replace(f.design, observation_id=f'planned-{i}-{n:g}', pattern=pattern, cycle_count=float(n)))
            result.append(tuple(designs))
        return tuple(result)


def parameter_space(*, mode='prior', dimension=2, unknown=('d0',), coefficients=None, nominal=None, shared_registration=False, load_unknown=False):
    nominal = nominal or {'A':8e-5, 'm':1.}
    specs = {'d0':Parameter('d0', .02, .30, '1'), 'A':Parameter('A', 1e-5, 5e-4, 'cycle^-1'), 'm':Parameter('m', .6, 1.6, '1')}
    specs.update({f'c{j}':Parameter(f'c{j}', -4., 4., '1') for j in range(dimension)})
    if mode in ('hybrid', 'direct') and coefficients is None:
        unknown = tuple(unknown)+tuple(f'c{j}' for j in range(dimension))
    fixed = {k:v for k,v in nominal.items() if k not in unknown}
    if coefficients is not None:
        if len(coefficients) != dimension or not np.isfinite(coefficients).all() or np.max(np.abs(coefficients)) > 4:
            raise ContractError('supplied correction coefficients disagree with the model')
        fixed.update({f'c{j}':float(a) for j,a in enumerate(coefficients)})
    if shared_registration:
        specs.update(tx=Parameter('tx', -.002, .002, 'mm'), ty=Parameter('ty', -.002, .002, 'mm'), theta=Parameter('theta', -.002, .002, 'rad'))
        unknown = tuple(unknown)+('tx','ty','theta')
    if load_unknown:
        specs['load_factor'] = Parameter('load_factor', .85, 1.15, '1')
        unknown = tuple(unknown)+('load_factor',)
    return ParameterSpace(tuple(specs[n] for n in unknown), fixed)


def public_patterns(metadata):
    m = metadata['mesh']; tags = {n:BoundaryTag(np.asarray(t['node_ids'], dtype=int), np.asarray(t['facets'], dtype=int).reshape(-1,2)) for n,t in m['boundary_tags'].items()}
    mesh = MeshData(m['nodes'], m['connectivity'], ['tri3']*len(m['connectivity']), np.asarray(m['material_region_ids'], dtype=int), 'mm', 'specimen', tags, {}, m['mesh_id'])
    mat = metadata['material']; materials = {int(r):MaterialAssignment(IsotropicPlaneStressMM(), {'young_modulus_n_per_mm2':mat['E'], 'poisson_ratio':mat['nu']}) for r in np.unique(mesh.material_region_ids)}
    left = mesh.boundary_tags['left'].node_ids; anchor = int(np.argmin(np.linalg.norm(mesh.nodes,axis=1)))
    bc = (DisplacementBC(left, np.zeros(len(left), dtype=int), np.zeros(len(left)), 'mm'), DisplacementBC([anchor],[1],[0.],'mm'))
    return tuple(Tri3SolverRequest.from_mm(mesh, materials, LoadingState(f'correction-load-{i}','static'), thickness_mm=mat['thickness'], displacement_bcs=bc, traction_bcs=(TractionBC('right',[10.*scale,0.],'N/mm^2'),)) for i,scale in enumerate(metadata['load_scales']))


def public_history(pattern, metadata):
    R = metadata['R']; end = metadata['assimilation_cutoff']; last = max(metadata['cycles']); f = metadata['future_load_factor']
    return (FatigueLoadCycle(0., end, 1., R, R, ('left','right'), LoadingMode.LOAD, load_id=pattern.loading_state.load_id),
            FatigueLoadCycle(end, last-end, f, f*R, R, ('left','right'), LoadingMode.LOAD, load_id=pattern.loading_state.load_id))


def load_correction_acquisition(folder, *, mode='prior', dimension=2, unknown=('d0',), coefficients=None,
                                included_cycles=None, cutoff=20., step=None, shared_registration=False,
                                load_unknown=False, law_spec=None, known_physics_override=None):
    path = Path(folder); meta = json.loads((path/'public.json').read_text())
    if set(meta) != PUBLIC_KEYS or meta['schema'] != 'phase2c-public-v1' or meta['units'] != {'length':'mm','stress':'N/mm^2','hazard':'cycle^-1','cycles':'cycle'}:
        raise ContractError('public correction schema contains unsupported or hidden channels')
    if cutoff > meta['assimilation_cutoff']:
        raise ContractError('requested assimilation cutoff exceeds the frozen contract')
    known = meta['nominal'] if known_physics_override is None else known_physics_override
    if set(known) != {'A', 'm'}:
        raise ContractError('explicit known-physics override can contain only A and m')
    space = parameter_space(mode=mode, dimension=dimension, unknown=unknown, coefficients=coefficients,
        nominal=known, shared_registration=shared_registration, load_unknown=load_unknown)
    predictors, batches = [], []
    for i, pattern in enumerate(public_patterns(meta)):
        spec = law_spec or LawSpecification(mode=mode, dimension=dimension)
        p = CorrectionPredictor(UniformDamage(), space, pattern, law_spec=spec,
            load_history=public_history(pattern, meta), cycle_step=step or meta['step'])
        frames = []
        for j,n in enumerate(meta['cycles']):
            if n > cutoff or included_cycles is not None and n not in included_cycles:
                continue
            measurement = DICMeasurement.load(path/'observations'/str(i)/str(j))
            diag_pattern = pattern if n <= meta['assimilation_cutoff'] else replace(pattern, traction_bcs=tuple(replace(bc, values=bc.values*meta['future_load_factor']) for bc in pattern.traction_bcs))
            frames.append(ObservationFrame(f"{meta['case_id']}-{i}-{j}", StaticLoadCase(diag_pattern, measurement, variance_floor_mm2=1e-14 if meta['sigma_mm']==0 else 0.)))
        contract = 'independent_given_shared_registration' if shared_registration else 'conditionally_independent'
        batches.append(ObservationBatch(tuple(frames), noise_contract=contract)); predictors.append(p)
    return CorrectionAcquisition(meta, tuple(predictors), tuple(batches))
