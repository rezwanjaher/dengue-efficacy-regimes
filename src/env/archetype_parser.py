"""Read a parameter set from configs/archetypes.json.
"""

import json
import numpy as np

class ArchetypeParser:
    def __init__(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def get_archetype_data(self, group: str, name: str):
        try:
            return self.data["archetype_groups"][group][name]
        except KeyError:
            raise ValueError(f"Archetype {group}/{name} not found.")

    def sample_parameters(self, archetype_data: dict, economic_scenario_name: str, rng: np.random.Generator):
        params = {}
        
        def parse_value(val):
            if isinstance(val, dict):
                if 'type' in val and val['type'] == 'uniform':
                    return float(rng.uniform(val['min'], val['max']))
                else:
                    return {k: parse_value(v) for k, v in val.items() if not k.startswith("_comment")}
            elif isinstance(val, list):
                return [parse_value(v) for v in val]
            else:
                return val

        eco_scenario = archetype_data["economic_scenarios"][economic_scenario_name]
        for key, value in eco_scenario.items():
            if not key.startswith("_comment"):
                params[key] = parse_value(value)

        bio_facts = archetype_data["biological_facts"]
        for key, value in bio_facts.items():
            if not key.startswith("_comment"):
                params[key] = parse_value(value)
        
        return params