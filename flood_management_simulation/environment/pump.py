# environment/pump.py

class Pump:
    def __init__(self, pump_id, max_capacity, energy_coefficient):
        self.pump_id = pump_id
        self.max_capacity = max_capacity  # Maximum water removal rate (m³/s)
        self.current_speed = 0.0  # Operational speed (m³/s)
        self.is_operational = True
        self.energy_coefficient = energy_coefficient  # Energy consumption per m³

    def set_speed(self, speed):
        """
        Set the pump speed within allowable limits.
        """
        if self.is_operational:
            self.current_speed = min(max(speed, 0.0), self.max_capacity)
        else:
            self.current_speed = 0.0

    def get_energy_consumption(self):
        """
        Calculate energy consumption based on current speed.
        """
        return self.current_speed * self.energy_coefficient

    def reset(self):
        """
        Reset pump to initial state.
        """
        self.current_speed = 0.0
        self.is_operational = True
