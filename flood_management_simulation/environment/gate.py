# environment/gate.py

class Gate:
    def __init__(self, gate_id, opening_level=1.0):
        self.gate_id = gate_id
        self.opening_level = opening_level  # Between 0.0 (closed) and 1.0 (fully open)
        self.is_operational = True

    def set_opening_level(self, level):
        """
        Set the gate opening level within allowable limits.
        """
        if self.is_operational:
            self.opening_level = min(max(level, 0.0), 1.0)
        else:
            self.opening_level = 0.0

    def reset(self):
        """
        Reset gate to initial state.
        """
        self.opening_level = 1.0
        self.is_operational = True
