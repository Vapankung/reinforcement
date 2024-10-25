# environment/zone.py

class Zone:
    def __init__(self, zone_id, capacity, flood_threshold, neighbors):
        self.zone_id = zone_id
        self.capacity = capacity
        self.flood_threshold = flood_threshold
        self.current_water_level = 0.0
        self.neighbors = neighbors
        self.inflow = 0.0
        self.outflow = 0.0
        self.is_flooded = False

    def update_water_level(self, delta_time):
        """
        Update water level based on inflow and outflow over a time step.
        """
        net_flow = (self.inflow - self.outflow) * delta_time
        self.current_water_level += net_flow
        self.current_water_level = max(self.current_water_level, 0.0)
        self.is_flooded = self.current_water_level > self.flood_threshold

    def reset(self):
        """
        Reset zone to initial state.
        """
        self.current_water_level = 0.0
        self.inflow = 0.0
        self.outflow = 0.0
        self.is_flooded = False
