import pygame
import numpy as np
import math
import random
import sys
from typing import List, Tuple, Optional
from dataclasses import dataclass

# ==============================================================================
# CONSTANTS AND CONFIGURATION
# ==============================================================================
WIDTH, HEIGHT = 1600, 900
FPS = 60
G = 1.5  # Gravitational constant
DT = 0.5  # Time step for integration
SOFTENING = 8.0  # Softening parameter to prevent singularity
TRAIL_LENGTH = 40  # Number of positions to keep in trail history
MAX_BODIES = 300  # Maximum number of bodies for performance
MIN_MASS = 10
MAX_SPAWN_MASS = 500
VELOCITY_SCALE = 0.02  # Scaling factor for mouse drag velocity

# Color palette for different mass ranges
COLOR_LOW_MASS = np.array([100, 150, 255])  # Blue
COLOR_MED_MASS = np.array([100, 255, 100])  # Green
COLOR_HIGH_MASS = np.array([255, 200, 100])  # Orange
COLOR_EXTREME_MASS = np.array([255, 100, 100])  # Red


class Body:
    """
    Represents an astronomical body with mass, position, velocity, and acceleration.
    Implements RK4 integration for stable orbital mechanics.
    """
    
    def __init__(self, x: float, y: float, mass: float, vx: float = 0.0, vy: float = 0.0):
        """
        Initialize a celestial body.
        
        Args:
            x, y: Initial position coordinates
            mass: Mass of the body (affects gravitational pull and visual size)
            vx, vy: Initial velocity components
        """
        self.pos = np.array([float(x), float(y)], dtype=np.float64)
        self.vel = np.array([float(vx), float(vy)], dtype=np.float64)
        self.acc = np.array([0.0, 0.0], dtype=np.float64)
        self.mass = float(mass)
        self.radius = self._calculate_radius()
        self.color = self._calculate_color()
        self.trail = []  # List of past positions for visualization
        self.alive = True
        
    def _calculate_radius(self) -> int:
        """
        Calculate visual radius based on mass.
        Assumes constant density: r ~ m^(1/3)
        """
        return max(3, int((self.mass ** 0.333) * 4))
    
    def _calculate_color(self) -> Tuple[int, int, int]:
        """
        Determine body color based on mass using smooth gradient interpolation.
        """
        if self.mass < 100:
            t = self.mass / 100.0
            color = (1 - t) * COLOR_LOW_MASS + t * COLOR_MED_MASS
        elif self.mass < 400:
            t = (self.mass - 100) / 300.0
            color = (1 - t) * COLOR_MED_MASS + t * COLOR_HIGH_MASS
        else:
            t = min(1.0, (self.mass - 400) / 600.0)
            color = (1 - t) * COLOR_HIGH_MASS + t * COLOR_EXTREME_MASS
        return tuple(color.astype(int))
    
    def compute_acceleration(self, bodies: List['Body']) -> np.ndarray:
        """
        Compute gravitational acceleration due to all other bodies.
        Uses Newton's law of gravitation with softening parameter.
        
        Args:
            bodies: List of all bodies in the simulation
            
        Returns:
            numpy array of acceleration vector [ax, ay]
        """
        acc = np.array([0.0, 0.0])
        for body in bodies:
            if body is self or not body.alive:
                continue
            r = body.pos - self.pos
            dist_sq = np.dot(r, r) + SOFTENING ** 2
            dist = math.sqrt(dist_sq)
            # F = G * m1 * m2 / r^2, a = F / m1 = G * m2 / r^2
            force_mag = G * body.mass / dist_sq
            acc += force_mag * (r / dist)
        return acc
    
    def update_rk4(self, bodies: List['Body'], dt: float):
        """
        Update position and velocity using 4th-order Runge-Kutta integration.
        This method provides high accuracy for orbital mechanics.
        
        Args:
            bodies: List of all bodies for force calculation
            dt: Time step
        """
        # Store initial state
        pos0 = self.pos.copy()
        vel0 = self.vel.copy()
        
        # k1 derivatives
        k1_v = self.compute_acceleration(bodies)
        k1_x = vel0
        
        # k2 derivatives (at t + dt/2 using k1)
        self.pos = pos0 + 0.5 * dt * k1_x
        k2_v = self.compute_acceleration(bodies)
        k2_x = vel0 + 0.5 * dt * k1_v
        
        # k3 derivatives (at t + dt/2 using k2)
        self.pos = pos0 + 0.5 * dt * k2_x
        k3_v = self.compute_acceleration(bodies)
        k3_x = vel0 + 0.5 * dt * k2_v
        
        # k4 derivatives (at t + dt using k3)
        self.pos = pos0 + dt * k3_x
        k4_v = self.compute_acceleration(bodies)
        k4_x = vel0 + dt * k3_v
        
        # Combine derivatives with RK4 weights
        self.pos = pos0 + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        self.vel = vel0 + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        # Update trail history
        self.trail.append(self.pos.copy())
        if len(self.trail) > TRAIL_LENGTH:
            self.trail.pop(0)
    
    def merge_with(self, other: 'Body'):
        """
        Inelastic collision: merge another body into this one.
        Conserves momentum: p_total = m1*v1 + m2*v2
        
        Args:
            other: Body to merge into this one
        """
        total_mass = self.mass + other.mass
        # Conservation of momentum: v_new = (m1*v1 + m2*v2) / (m1 + m2)
        self.vel = (self.vel * self.mass + other.vel * other.mass) / total_mass
        self.mass = total_mass
        self.radius = self._calculate_radius()
        self.color = self._calculate_color()
        other.alive = False
    
    def draw(self, surface: pygame.Surface, offset_x: int = 0, offset_y: int = 0):
        """
        Render the body and its orbital trail.
        
        Args:
            surface: Pygame surface to draw on
        """
        # Draw motion trail with fading opacity
        if len(self.trail) > 1:
            points = [(int(p[0] + offset_x), int(p[1] + offset_y)) for p in self.trail]
            for i in range(len(points) - 1):
                alpha = int(255 * (i / len(points)) * 0.6)
                if alpha > 0:
                    color = (max(0, self.color[0] - 50), 
                            max(0, self.color[1] - 50), 
                            max(0, self.color[2] - 50))
                    pygame.draw.line(surface, color, points[i], points[i+1], 
                                   max(1, int(self.radius * 0.3)))
        
        # Draw glow for massive bodies
        if self.mass > 200:
            glow_radius = self.radius * 3
            glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            for r in range(glow_radius, 0, -2):
                alpha = int(20 * (1 - r / glow_radius))
                pygame.draw.circle(glow_surf, (*self.color, alpha), 
                                 (glow_radius, glow_radius), r)
            surface.blit(glow_surf, (int(self.pos[0] - glow_radius + offset_x), 
                                    int(self.pos[1] - glow_radius + offset_y)))
        
        # Draw main body
        pygame.draw.circle(surface, self.color, 
                          (int(self.pos[0] + offset_x), int(self.pos[1] + offset_y)), 
                          self.radius)
        # Highlight
        pygame.draw.circle(surface, (255, 255, 255), 
                          (int(self.pos[0] - self.radius*0.3 + offset_x), 
                           int(self.pos[1] - self.radius*0.3 + offset_y)), 
                          max(1, self.radius // 4))


class Starfield:
    """
    Background star field with parallax depth effect.
    """
    def __init__(self, count: int = 300):
        self.stars = []
        for _ in range(count):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            brightness = random.randint(30, 200)
            size = random.choice([1, 1, 1, 2])  # Mostly small stars
            self.stars.append((x, y, brightness, size))
    
    def draw(self, surface: pygame.Surface, offset_x: int = 0, offset_y: int = 0):
        """Draw static starfield background."""
        for x, y, brightness, size in self.stars:
            color = (brightness, brightness, brightness + 20)
            px = (x + offset_x * 0.1) % WIDTH
            py = (y + offset_y * 0.1) % HEIGHT
            if size == 1:
                surface.set_at((int(px), int(py)), color)
            else:
                pygame.draw.circle(surface, color, (int(px), int(py)), size)


class UI:
    """
    User interface overlay for simulation statistics and controls.
    """
    def __init__(self, font: pygame.font.Font):
        self.font = font
        self.small_font = pygame.font.SysFont('monospace', 12)
    
    def draw(self, surface: pygame.Surface, bodies: List[Body], fps: float, 
             spawning: bool, spawn_mass: float):
        """
        Draw information overlay.
        
        Args:
            surface: Target surface
            bodies: Current list of bodies
            fps: Current frames per second
            spawning: Whether user is currently spawning a body
            spawn_mass: Mass of body being spawned
        """
        # Semi-transparent background panel
        panel = pygame.Surface((350, 150))
        panel.set_alpha(180)
        panel.fill((20, 20, 30))
        surface.blit(panel, (10, 10))
        
        # Statistics
        total_mass = sum(b.mass for b in bodies)
        kinetic_energy = sum(0.5 * b.mass * np.dot(b.vel, b.vel) for b in bodies)
        
        lines = [
            f"N-Body Simulation",
            f"Bodies: {len(bodies)} / {MAX_BODIES}",
            f"Total Mass: {total_mass:.0f}",
            f"Kinetic Energy: {kinetic_energy:.0f}",
            f"FPS: {fps:.1f}",
            "",
            "Controls:",
            "LMB + Drag: Create body (drag for velocity)",
            "RMB + Drag: Pan camera",
            "Hold longer: Increase mass",
            "R: Reset | ESC: Quit"
        ]
        
        if spawning:
            lines.append(f"Spawning Mass: {spawn_mass:.0f}")
        
        y_offset = 20
        for line in lines:
            if line.startswith("N-Body") or line.startswith("Controls"):
                text = self.font.render(line, True, (255, 220, 150))
            else:
                text = self.small_font.render(line, True, (200, 200, 200))
            surface.blit(text, (20, y_offset))
            y_offset += 18


class Simulation:
    """
    Main simulation controller handling physics, rendering, and user interaction.
    """
    
    def __init__(self):
        """Initialize Pygame and simulation state."""
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Professional N-Body Gravitational Simulation")
        self.clock = pygame.time.Clock()
        
        # Fonts
        try:
            self.font = pygame.font.SysFont('monospace', 16, bold=True)
        except:
            self.font = pygame.font.Font(None, 20)
        
        # Initialize components
        self.starfield = Starfield()
        self.ui = UI(self.font)
        self.bodies: List[Body] = []
        
        # Spawning state
        self.spawning = False
        self.spawn_start_pos: Optional[Tuple[int, int]] = None
        self.spawn_start_time = 0
        
        # Camera state
        self.camera_offset = [0.0, 0.0]
        self.panning = False
        self.pan_start_pos: Optional[Tuple[int, int]] = None
        self.camera_start_offset = [0.0, 0.0]
        
        # Create initial system (binary star with planets)
        self._initialize_system()
    
    def _initialize_system(self):
        """Create an interesting initial configuration."""
        self.bodies = []
        
        # Binary star system
        star1 = Body(WIDTH/2 - 100, HEIGHT/2, 800, 0, 0.8)
        star2 = Body(WIDTH/2 + 100, HEIGHT/2, 600, 0, -1.0)
        
        self.bodies.append(star1)
        self.bodies.append(star2)
        
        # Add some random planets
        for _ in range(5):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(200, 400)
            x = WIDTH/2 + math.cos(angle) * dist
            y = HEIGHT/2 + math.sin(angle) * dist
            
            # Orbital velocity perpendicular to radius vector
            v_mag = math.sqrt(G * 1400 / dist)  # Approximate total central mass
            vx = -math.sin(angle) * v_mag * random.uniform(0.8, 1.2)
            vy = math.cos(angle) * v_mag * random.uniform(0.8, 1.2)
            
            mass = random.uniform(20, 80)
            self.bodies.append(Body(x, y, mass, vx, vy))
    
    def handle_events(self) -> bool:
        """
        Process user input events.
        Returns False if simulation should quit.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self._initialize_system()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and len(self.bodies) < MAX_BODIES:  # Left click
                    self.spawning = True
                    self.spawn_start_pos = event.pos
                    self.spawn_start_time = pygame.time.get_ticks()
                elif event.button == 3:  # Right click
                    self.panning = True
                    self.pan_start_pos = event.pos
                    self.camera_start_offset = list(self.camera_offset)
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.spawning:
                    self._spawn_body(event.pos)
                    self.spawning = False
                elif event.button == 3:
                    self.panning = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.panning and self.pan_start_pos:
                    dx = event.pos[0] - self.pan_start_pos[0]
                    dy = event.pos[1] - self.pan_start_pos[1]
                    self.camera_offset[0] = self.camera_start_offset[0] + dx
                    self.camera_offset[1] = self.camera_start_offset[1] + dy
        
        return True
    
    def _spawn_body(self, end_pos: Tuple[int, int]):
        """
        Create new body based on mouse drag.
        Velocity is determined by drag vector, mass by hold duration.
        """
        if not self.spawn_start_pos:
            return
        
        
        dx = (self.spawn_start_pos[0] - end_pos[0]) * VELOCITY_SCALE
        dy = (self.spawn_start_pos[1] - end_pos[1]) * VELOCITY_SCALE
        
        
        hold_time = pygame.time.get_ticks() - self.spawn_start_time
        mass = MIN_MASS + min(MAX_SPAWN_MASS, hold_time / 2)
        
        
        x = self.spawn_start_pos[0] - self.camera_offset[0]
        y = self.spawn_start_pos[1] - self.camera_offset[1]
        body = Body(x, y, mass, dx, dy)
        self.bodies.append(body)
    
    def resolve_collisions(self):
        """
        Detect and resolve collisions between bodies.
        Uses simple distance-based merging (inelastic collision).
        """
        to_remove = set()
        
        for i in range(len(self.bodies)):
            if i in to_remove:
                continue
            body1 = self.bodies[i]
            
            for j in range(i + 1, len(self.bodies)):
                if j in to_remove:
                    continue
                body2 = self.bodies[j]
                
                # Distance check
                dist_vec = body1.pos - body2.pos
                dist = math.sqrt(dist_vec[0]**2 + dist_vec[1]**2)
                
                if dist < (body1.radius + body2.radius) * 0.8:
                    # Merge smaller into larger
                    if body1.mass >= body2.mass:
                        body1.merge_with(body2)
                        to_remove.add(j)
                    else:
                        body2.merge_with(body1)
                        to_remove.add(i)
                        break
        
        # Remove merged bodies
        if to_remove:
            self.bodies = [b for i, b in enumerate(self.bodies) if i not in to_remove]
    
    def update(self):
        """Update physics for all bodies."""
        # RK4 integration step for all bodies
        for body in self.bodies:
            body.update_rk4(self.bodies, DT)
        
        
        margin = 500
        cam_x, cam_y = self.camera_offset
        self.bodies = [
            b for b in self.bodies 
            if (-margin < b.pos[0] + cam_x < WIDTH + margin and 
                -margin < b.pos[1] + cam_y < HEIGHT + margin)
        ]
        
        # Handle collisions
        self.resolve_collisions()
    
    def draw(self):
        """Render the simulation frame."""
        # Clear with deep space color
        self.screen.fill((5, 5, 15))
        
        # Draw background
        cam_x, cam_y = int(self.camera_offset[0]), int(self.camera_offset[1])
        self.starfield.draw(self.screen, cam_x, cam_y)
        
        # Draw bodies
        for body in self.bodies:
            body.draw(self.screen, cam_x, cam_y)
        
        # Draw spawn preview
        if self.spawning and self.spawn_start_pos:
            mouse_pos = pygame.mouse.get_pos()
            
            
            pygame.draw.line(self.screen, (200, 200, 200), 
                           self.spawn_start_pos, mouse_pos, 2)
            
            
            hold_time = pygame.time.get_ticks() - self.spawn_start_time
            mass = MIN_MASS + min(MAX_SPAWN_MASS, hold_time / 2)
            radius = max(3, int((mass ** 0.333) * 4))
            
            
            pygame.draw.circle(self.screen, (200, 200, 200), 
                             self.spawn_start_pos, radius, 1)
            
            
            pygame.draw.circle(self.screen, (255, 255, 255), mouse_pos, 3)
        
        # Draw UI
        fps = self.clock.get_fps()
        hold_time = pygame.time.get_ticks() - self.spawn_start_time if self.spawning else 0
        spawn_mass = MIN_MASS + min(MAX_SPAWN_MASS, hold_time / 2)
        self.ui.draw(self.screen, self.bodies, fps, self.spawning, spawn_mass)
        
        pygame.display.flip()
    
    def run(self):
        """Main simulation loop."""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    sim = Simulation()
    sim.run()