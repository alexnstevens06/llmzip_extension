import math

G = 6.674e-11
DT = 0.01

class Body:
    def __init__(self, mass, x, y, vx, vy):
        self.mass = mass
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

def compute_force(b1, b2):
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1e-10:
        return 0.0, 0.0
    f = G * b1.mass * b2.mass / (dist * dist)
    fx = f * dx / dist
    fy = f * dy / dist
    return fx, fy

def step(bodies):
    forces = {}
    for i, b1 in enumerate(bodies):
        fx_total = 0.0
        fy_total = 0.0
        for j, b2 in enumerate(bodies):
            if i != j:
                fx, fy = compute_force(b1, b2)
                fx_total += fx
                fy_total += fy
        forces[i] = (fx_total, fy_total)
    for i, body in enumerate(bodies):
        ax = forces[i][0] / body.mass
        ay = forces[i][1] / body.mass
        body.vx += ax * DT
        body.vy += ay * DT
        body.x += body.vx * DT
        body.y += body.vy * DT

def kinetic_energy(bodies):
    ke = 0.0
    for b in bodies:
        ke += 0.5 * b.mass * (b.vx**2 + b.vy**2)
    return ke

def potential_energy(bodies):
    pe = 0.0
    for i in range(len(bodies)):
        for j in range(i+1, len(bodies)):
            dx = bodies[j].x - bodies[i].x
            dy = bodies[j].y - bodies[i].y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 1e-10:
                pe -= G * bodies[i].mass * bodies[j].mass / dist
    return pe

def simulate(bodies, steps):
    for s in range(steps):
        step(bodies)
        if s % 100 == 0:
            ke = kinetic_energy(bodies)
            pe = potential_energy(bodies)
            print(f"Step {s}: KE={ke:.6e}, PE={pe:.6e}, Total={ke+pe:.6e}")

if __name__ == "__main__":
    sun = Body(1.989e30, 0, 0, 0, 0)
    earth = Body(5.972e24, 1.496e11, 0, 0, 29783)
    moon = Body(7.342e22, 1.496e11 + 3.844e8, 0, 0, 29783 + 1022)
    bodies = [sun, earth, moon]
    simulate(bodies, 1000)
    for b in bodies:
        print(f"Mass={b.mass:.3e}, Pos=({b.x:.3e}, {b.y:.3e})")
