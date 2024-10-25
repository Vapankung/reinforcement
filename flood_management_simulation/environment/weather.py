# environment/weather.py

import numpy as np

class Weather:
    def __init__(self, base_intensity=5.0, variance=2.0, extreme_intensity=20.0, extreme_probability=0.1):
        self.base_intensity = base_intensity
        self.variance = variance
        self.extreme_intensity = extreme_intensity
        self.extreme_probability = extreme_probability

    def generate_rainfall(self, zones):
        """
        Generate rainfall for each zone based on normal and extreme conditions.

        Parameters:
        - zones: Iterable of zone IDs.

        Returns:
        - rainfall: Dictionary mapping zone IDs to rainfall intensities (mm/hour).
        """
        rainfall = {}
        for zone_id in zones:
            if np.random.rand() < self.extreme_probability:
                intensity = np.random.normal(self.extreme_intensity, self.variance)
            else:
                intensity = np.random.normal(self.base_intensity, self.variance)
            intensity = max(intensity, 0.0)  # Ensure non-negative rainfall
            rainfall[zone_id] = intensity  # in mm/hour
        return rainfall
