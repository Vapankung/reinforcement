# visualization/visualization.py

import pygame
import sys
import random

class RainParticle:
    def __init__(self, x, y, speed, length, color):
        self.x = x
        self.y = y
        self.speed = speed
        self.length = length
        self.color = color

    def update(self):
        self.y += self.speed

    def draw(self, surface):
        pygame.draw.line(surface, self.color, (self.x, self.y), (self.x, self.y + self.length), 1)

class Visualization:
    def __init__(self, zones):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('Flood Management Simulation')
        self.zones = zones
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14)
        self.zone_positions = self.calculate_positions()
        self.rain_particles = []
        print("Pygame Visualization Initialized.")

    def calculate_positions(self):
        """
        Calculate screen positions for each zone.

        Returns:
        - positions: Dictionary mapping zone IDs to (x, y) coordinates.
        """
        positions = {}
        margin = 50
        zone_width = (800 - 2 * margin) // 4
        zone_height = (600 - 2 * margin) // 2
        for idx, zone_id in enumerate(self.zones):
            x = margin + (idx % 4) * zone_width + zone_width // 2
            y = margin + (idx // 4) * zone_height + zone_height // 2
            positions[zone_id] = (x, y)
        return positions

    def update(self, rainfall):
        """
        Update the visualization by rendering the current state and rain particles.

        Parameters:
        - rainfall: Dictionary mapping zone IDs to rainfall intensities (mm/hour).
        """
        print("Visualization Update Called.")
        print(f"Rainfall Data: {rainfall}")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((30, 30, 30))  # Dark background

        # Draw zones
        for zone_id, zone in self.zones.items():
            x, y = self.zone_positions[zone_id]
            water_level_ratio = zone.current_water_level / zone.capacity
            color = self.get_color(water_level_ratio)
            rect = pygame.Rect(x - 75, y - 50, 150, 100)  # Centered rectangle
            pygame.draw.rect(self.screen, color, rect)
            text_surface = self.font.render(f'{zone_id}', True, (255, 255, 255))
            self.screen.blit(text_surface, (x - 40, y - 45))
            water_text = self.font.render(f'Water: {zone.current_water_level:.2f}', True, (255, 255, 255))
            self.screen.blit(water_text, (x - 40, y - 25))

            if zone.is_flooded:
                flood_text = self.font.render('Flooded!', True, (255, 0, 0))
                self.screen.blit(flood_text, (x - 40, y - 5))

        # Update and draw rain particles
        self.spawn_rain_particles(rainfall)
        self.update_rain_particles()
        self.draw_rain_particles()

        pygame.display.flip()
        self.clock.tick(60)  # Limit to 60 FPS

    def get_color(self, water_level_ratio):
        """
        Determine the color of a zone based on its water level ratio.

        Parameters:
        - water_level_ratio: Float representing the normalized water level.

        Returns:
        - color: Tuple representing the RGB color.
        """
        if water_level_ratio < 0.5:
            return (0, 128, 0)  # Dark green
        elif water_level_ratio < 0.8:
            return (255, 255, 0)  # Yellow
        elif water_level_ratio < 1.0:
            return (255, 165, 0)  # Orange
        else:
            return (255, 0, 0)    # Red

    def spawn_rain_particles(self, rainfall):
        """
        Spawn rain particles based on the current rainfall intensity.

        Parameters:
        - rainfall: Dictionary mapping zone IDs to rainfall intensities (mm/hour).
        """
        for zone_id, intensity in rainfall.items():
            # Determine number of particles based on intensity
            num_particles = int(intensity / 2)  # Adjust divisor for suitable number
            x, y = self.zone_positions[zone_id]
            for _ in range(num_particles):
                offset_x = random.randint(-75, 75)
                particle_x = x + offset_x
                particle_y = y - 60  # Start above the zone
                speed = random.uniform(5, 10)  # Speed of rain particle
                length = random.randint(5, 15)
                color = (138, 43, 226)  # Blue-ish color for rain
                self.rain_particles.append(RainParticle(particle_x, particle_y, speed, length, color))

    def update_rain_particles(self):
        """
        Update the position of rain particles and remove those that are off-screen.
        """
        for particle in self.rain_particles[:]:
            particle.update()
            if particle.y > 600:
                self.rain_particles.remove(particle)

    def draw_rain_particles(self):
        """
        Draw all rain particles on the screen.
        """
        for particle in self.rain_particles:
            particle.draw(self.screen)

    def close(self):
        """
        Properly close the Pygame window.
        """
        pygame.quit()
