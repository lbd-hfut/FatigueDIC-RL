"""N0 observation-only inputs for uniform initial-damage cost probes.

The externally known healthy condition certifies D=0, not equality of the two
specimens' moduli. The current comparison deliberately assumes equal moduli;
its transfer bias is evaluated separately, never read by this loader.
"""
from pathlib import Path
import json
import numpy as np
from fatigue_dic_rl.core.validation import ContractError
from fatigue_dic_rl.correction.acquisition import public_patterns
from fatigue_dic_rl.inverse.forward_adapter import StaticLoadCase, ReactionObservation
from fatigue_dic_rl.observations.noise import DICMeasurement


PUBLIC_KEYS = {"schema", "case_id", "split", "mesh", "material", "load_scales",
               "healthy_certificate", "noise_contract", "units"}


def load_static_cases(folder, *, reaction=False, healthy=False):
    path = Path(folder)
    meta = json.loads((path/"public.json").read_text())
    if set(meta) != PUBLIC_KEYS or meta["schema"] != "state-readiness-N0-v1":
        raise ContractError("invalid N0 observation-only contract or unexpected truth channel")
    if meta["split"] != "development" or meta["units"] != {"length": "mm", "force": "N", "modulus": "N/mm^2"}:
        raise ContractError("this cost probe supports physical units and Development only")
    if healthy and meta["healthy_certificate"] != {"externally_known_damage": 0., "source": "independent_external_condition"}:
        raise ContractError("healthy D=0 requires an independent external condition")
    patterns = public_patterns(meta)
    cases = []
    for kind in ("damaged", "healthy") if healthy else ("damaged",):
        for i, pattern in enumerate(patterns):
            measurement = DICMeasurement.load(path/kind/str(i)/"dic")
            if measurement.observation.cycle_count != 0:
                raise ContractError("initial-state readiness cannot read future fatigue observations")
            force = None
            if reaction and kind == "damaged":
                data = json.loads((path/kind/str(i)/"reaction.json").read_text())
                if set(data) != {"value_n", "standard_deviation_n", "node_ids", "component"}:
                    raise ContractError("unexpected reaction input channels")
                force = ReactionObservation(np.asarray(data["node_ids"]), data["component"],
                    data["value_n"], data["standard_deviation_n"])
            cases.append(StaticLoadCase(pattern, measurement, force, known_healthy=kind == "healthy"))
    return tuple(cases)
