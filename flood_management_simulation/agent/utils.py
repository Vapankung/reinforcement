# agent/utils.py

from itertools import product

def generate_discrete_actions(pumps, gates):
    """
    Generate a list of all possible discrete actions by combining pump speeds and gate openings.
    Each action adjusts one pump and one gate at a time.

    Parameters:
    - pumps: Dictionary of Pump objects.
    - gates: Dictionary of Gate objects.

    Returns:
    - actions: List of action dictionaries.
    """
    pump_levels = [0.0, 0.5, 1.0]  # Reduced discrete levels for pumps
    gate_levels = [0.0, 0.5, 1.0]  # Discrete levels for gates

    pump_ids = list(pumps.keys())
    gate_ids = list(gates.keys())

    actions = []
    for pump_id, p_level in product(pump_ids, pump_levels):
        for gate_id, g_level in product(gate_ids, gate_levels):
            action = {
                'pumps': {pump_id: p_level * pumps[pump_id].max_capacity},  # Only one pump adjusted
                'gates': {gate_id: g_level}  # Only one gate adjusted
            }
            actions.append(action)
    return actions
