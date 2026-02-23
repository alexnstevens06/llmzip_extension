import math
from dataclasses import dataclass, field
from typing import List, Tuple

# Physical constants
GRAVITATIONAL_CONSTANT = 6.674e-11  # m^3 kg^-1 s^-2
SOFTENING_EPSILON = 1e-10  # Prevent division by zero in force calculation

@dataclass
class CelestialBody:
    """Represents a body in the N-body gravitational simulation."""
    name: str
    mass: float  # kilograms
    x: float = 0.0  # meters
    y: float = 0.0
    vx: float = 0.0  # meters/second
    vy: float = 0.0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    def kinetic_energy(self) -> float:
        """Calculate the kinetic energy of this body: KE = 0.5 * m * v^2"""
        return 0.5 * self.mass * (self.vx ** 2 + self.vy ** 2)

    def record_position(self) -> None:
        """Save current position to trajectory history."""
        self.trajectory.append((self.x, self.y))

def compute_gravitational_force(
    body1: CelestialBody, body2: CelestialBody
) -> Tuple[float, float]:
    """
    Calculate the gravitational force vector exerted on body1 by body2.
    Uses softening parameter to avoid singularity at zero separation.
    
    Returns:
        Tuple of (fx, fy) force components in Newtons.
    """
    dx = body2.x - body1.x
    dy = body2.y - body1.y
    distance_squared = dx * dx + dy * dy
    distance = math.sqrt(distance_squared)
    if distance < SOFTENING_EPSILON:
        return 0.0, 0.0
    force_magnitude = (
        GRAVITATIONAL_CONSTANT * body1.mass * body2.mass / distance_squared
    )
    fx = force_magnitude * dx / distance
    fy = force_magnitude * dy / distance
    return fx, fy

def leapfrog_step(bodies: List[CelestialBody], dt: float) -> None:
    """
    Advance the simulation by one timestep using the leapfrog integrator.
    This is a symplectic integrator that conserves energy better than Euler.
    """
    n = len(bodies)
    accelerations = []
    # Calculate all accelerations based on current positions
    for i in range(n):
        ax_total, ay_total = 0.0, 0.0
        for j in range(n):
            if i != j:
                fx, fy = compute_gravitational_force(bodies[i], bodies[j])
                ax_total += fx / bodies[i].mass
                ay_total += fy / bodies[i].mass
        accelerations.append((ax_total, ay_total))
    # Update velocities and positions using leapfrog scheme
    for i, body in enumerate(bodies):
        ax, ay = accelerations[i]
        body.vx += ax * dt
        body.vy += ay * dt
        body.x += body.vx * dt
        body.y += body.vy * dt
        body.record_position()

def total_kinetic_energy(bodies: List[CelestialBody]) -> float:
    """Sum of kinetic energies of all bodies."""
    return sum(b.kinetic_energy() for b in bodies)

def total_potential_energy(bodies: List[CelestialBody]) -> float:
    """Sum of gravitational potential energy for all unique pairs."""
    pe = 0.0
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            dx = bodies[j].x - bodies[i].x
            dy = bodies[j].y - bodies[i].y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > SOFTENING_EPSILON:
                pe -= (
                    GRAVITATIONAL_CONSTANT
                    * bodies[i].mass
                    * bodies[j].mass
                    / dist
                )
    return pe

def simulate(
    bodies: List[CelestialBody],
    num_steps: int,
    dt: float = 0.01,
    report_interval: int = 100,
) -> dict:
    """
    Run the N-body simulation and return energy conservation metrics.
    
    Args:
        bodies: List of celestial bodies to simulate.
        num_steps: Number of timesteps to advance.
        dt: Timestep in seconds.
        report_interval: Print energy report every N steps.
    
    Returns:
        Dictionary containing initial and final total energy.
    """
    initial_energy = total_kinetic_energy(bodies) + total_potential_energy(bodies)
    for step in range(num_steps):
        leapfrog_step(bodies, dt)
        if step % report_interval == 0:
            ke = total_kinetic_energy(bodies)
            pe = total_potential_energy(bodies)
            total = ke + pe
            drift = abs(total - initial_energy) / abs(initial_energy) * 100
            print(
                f"Step {step:6d}: KE={ke:.6e}, PE={pe:.6e}, "
                f"Total={total:.6e}, Drift={drift:.4f}%"
            )
    final_energy = total_kinetic_energy(bodies) + total_potential_energy(bodies)
    return {"initial_energy": initial_energy, "final_energy": final_energy}

if __name__ == "__main__":
    sun = CelestialBody("Sun", 1.989e30, 0, 0, 0, 0)
    earth = CelestialBody("Earth", 5.972e24, 1.496e11, 0, 0, 29783)
    moon = CelestialBody(
        "Moon", 7.342e22, 1.496e11 + 3.844e8, 0, 0, 29783 + 1022
    )
    bodies = [sun, earth, moon]
    results = simulate(bodies, num_steps=1000, dt=0.01, report_interval=200)
    print(f"\nEnergy conservation: {results}")
    for body in bodies:
        print(f"{body.name}: pos=({body.x:.3e}, {body.y:.3e}), "
              f"trajectory_points={len(body.trajectory)}")
