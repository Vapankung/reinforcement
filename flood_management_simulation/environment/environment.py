# environment/environment.py

import numpy as np
from .zone import Zone
from .pump import Pump
from .gate import Gate
from .weather import Weather
import random

class Environment:
    def __init__(self, delta_time):
        self.zones = {}
        self.pumps = {}
        self.gates = {}
        self.flow_coefficients = {}
        self.delta_time = delta_time  # in seconds
        self.weather = Weather()
        self.current_rainfall = {}
        self.setup_environment()

    def setup_environment(self):
        """
        Initialize zones, pumps, gates, and flow coefficients.
        """
        # Define 8 zones with random capacities and neighbors
        for i in range(8):
            zone_id = f'Zone_{i+1}'
            capacity = random.uniform(500, 1000)  # m³
            flood_threshold = capacity * 0.8
            neighbors = []
            if i > 0:
                neighbors.append(f'Zone_{i}')  # Previous zone
            if i < 7:
                neighbors.append(f'Zone_{i+2}')  # Next zone
            self.zones[zone_id] = Zone(zone_id, capacity, flood_threshold, neighbors)
            # Initialize pumps
            self.pumps[zone_id] = Pump(pump_id=zone_id, max_capacity=50.0, energy_coefficient=0.1)

        # Define gates between consecutive zones
        for i in range(7):
            gate_id = f'Gate_{i+1}'
            zone_i = f'Zone_{i+1}'
            zone_j = f'Zone_{i+2}'
            self.gates[(zone_i, zone_j)] = Gate(gate_id=gate_id)

        # Assign flow coefficients between zones
        for (zone_i, zone_j) in self.gates.keys():
            self.flow_coefficients[(zone_i, zone_j)] = random.uniform(0.5, 1.0)

    def compute_water_flows(self):
        """
        Compute water flows between zones based on hydraulic gradients and gate positions.
        """
        flows = {}
        for (zone_i_id, zone_j_id), coeff in self.flow_coefficients.items():
            zone_i = self.zones[zone_i_id]
            zone_j = self.zones[zone_j_id]
            water_level_diff = zone_i.current_water_level - zone_j.current_water_level

            gate = self.gates.get((zone_i_id, zone_j_id)) or self.gates.get((zone_j_id, zone_i_id))
            if gate and not gate.is_operational:
                opening_level = 0.0
            else:
                opening_level = gate.opening_level if gate else 1.0

            flow_rate = coeff * opening_level * np.sqrt(abs(water_level_diff))
            flow_rate *= np.sign(water_level_diff)

            if water_level_diff == 0:
                flow_rate = 0.0

            flows[(zone_i_id, zone_j_id)] = flow_rate

        return flows

    def update_infrastructure(self, action):
        """
        Update pump speeds and gate opening levels based on the action.

        Parameters:
        - action: Dictionary containing pump speeds and gate openings.
        """
        for pump_id, speed in action.get('pumps', {}).items():
            if pump_id in self.pumps:
                self.pumps[pump_id].set_speed(speed)

        for gate_key, opening_level in action.get('gates', {}).items():
            if gate_key in self.gates:
                self.gates[gate_key].set_opening_level(opening_level)

    def step(self, action):
        """
        Advance the simulation by one time step.

        Parameters:
        - action: Dictionary containing pump speeds and gate opening levels.
        """
        self.update_infrastructure(action)
        flows = self.compute_water_flows()
        rainfall = self.weather.generate_rainfall(self.zones.keys())
        self.current_rainfall = rainfall  # Store current rainfall for visualization

        for zone_id, zone in self.zones.items():
            zone.inflow = rainfall.get(zone_id, 0.0)  # Rainfall input
            zone.outflow = 0.0

            pump = self.pumps.get(zone_id)
            if pump and pump.is_operational:
                zone.outflow += pump.current_speed

            for neighbor_id in zone.neighbors:
                flow_in = flows.get((neighbor_id, zone_id), 0.0)
                flow_out = flows.get((zone_id, neighbor_id), 0.0)
                zone.inflow += max(flow_in, 0.0)
                zone.outflow += max(flow_out, 0.0)

            zone.update_water_level(self.delta_time)

    def get_state(self):
        """
        Get the current state as a numpy array.
        Includes normalized water levels, flood status, pump speeds, pump statuses, gate openings, and gate statuses.

        Returns:
        - state: Numpy array representing the current state.
        """
        state = []
        for zone in self.zones.values():
            state.append(zone.current_water_level / zone.capacity)
            state.append(float(zone.is_flooded))
        for pump in self.pumps.values():
            state.append(pump.current_speed / pump.max_capacity)
            state.append(float(pump.is_operational))
        for gate in self.gates.values():
            state.append(gate.opening_level)
            state.append(float(gate.is_operational))
        return np.array(state, dtype=np.float32)

    def get_current_rainfall(self):
        """
        Retrieve the latest rainfall data.

        Returns:
        - rainfall: Dictionary mapping zone IDs to rainfall intensities (mm/hour).
        """
        return self.current_rainfall

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        for zone in self.zones.values():
            zone.reset()
        for pump in self.pumps.values():
            pump.reset()
        for gate in self.gates.values():
            gate.reset()
        self.current_rainfall = {}
