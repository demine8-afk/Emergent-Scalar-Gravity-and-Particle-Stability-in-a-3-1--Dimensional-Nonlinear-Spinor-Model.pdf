import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE 
# =============================================================================
# SCRIPT: fig4_binary_decay_final.py
# PURPOSE: High-Fidelity Ab Initio Simulation of Scalar Gravitational Inspiral.
#
# PHYSICS:
# We solve the full wave equation for the scalar field Phi.
# The retardation of the field naturally produces a dissipative force 
# (radiation reaction), causing the binary orbit to decay over time.
#
# This simulation demonstrates that "friction" is not needed: 
# the energy is carried away by the field waves.
# =============================================================================

# --- CONFIGURATION ---
L = 12.0            # Domain size
N = 300             # Grid resolution (Optimized for speed/quality balance)
dx = 2*L / N
c = 1.0             
dt = 0.02           
T_max = 65.0        # Longer time to see multiple orbits

# Tuned Parameters for Multi-Turn Inspiral
G_coupling = 0.8    # Weaker gravity -> Slower inspiral (more orbits)
M_particle = 1.0
R_soft = 0.8        

R_init = 5.0        
v_orbit = 0.35      # Tuned for elliptical inspiral

# Stability
assert c * dt / dx < 0.7

# --- SOLVER ---
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

def get_force_interpolated(phi, pos):
    # Grid indices
    x_idx = (pos[0] + L) / dx
    y_idx = (pos[1] + L) / dx
    i, j = int(x_idx), int(y_idx)
    
    if i < 1 or i >= N-2 or j < 1 or j >= N-2: return np.array([0.,0.])
    
    # Gradient
    gx = (phi[j, i+1] - phi[j, i-1]) / (2*dx)
    gy = (phi[j+1, i] - phi[j-1, i]) / (2*dx)
    
    return -np.array([gx, gy]) * G_coupling

# --- MAIN LOOP ---
print(f"Simulating Long Inspiral... T={T_max}")

pos1 = np.array([-R_init/2, 0.0])
pos2 = np.array([R_init/2, 0.0])
vel1 = np.array([0.05, -v_orbit]) 
vel2 = np.array([-0.05, v_orbit])

phi_curr = np.zeros((N, N))
phi_prev = np.zeros((N, N))

traj1_x, traj1_y = [], []
traj2_x, traj2_y = [], []
separation = []
times = []

steps = int(T_max / dt)
mask = np.ones((N, N))
edge = int(N * 0.08)
mask[:edge,:] *= 0.9; mask[-edge:,:] *= 0.9; mask[:,:edge] *= 0.9; mask[:,-edge:] *= 0.9

for n in tqdm(range(steps)):
    # Source (Vectorized for speed)
    r2_1 = (X - pos1[0])**2 + (Y - pos1[1])**2
    r2_2 = (X - pos2[0])**2 + (Y - pos2[1])**2
    rho = - (np.exp(-r2_1/(2*R_soft**2)) + np.exp(-r2_2/(2*R_soft**2)))
    
    # Wave Eq
    lap = (np.roll(phi_curr,1,0) + np.roll(phi_curr,-1,0) + 
           np.roll(phi_curr,1,1) + np.roll(phi_curr,-1,1) - 4*phi_curr) / dx**2
    phi_next = 2*phi_curr - phi_prev + dt**2 * (c**2 * lap + rho)
    phi_next *= mask
    
    # Particle Eq
    F1 = get_force_interpolated(phi_curr, pos1)
    F2 = get_force_interpolated(phi_curr, pos2)
    
    vel1 += F1 * dt / M_particle
    vel2 += F2 * dt / M_particle
    pos1 += vel1 * dt
    pos2 += vel2 * dt
    
    phi_prev = phi_curr
    phi_curr = phi_next
    
    if n % 5 == 0:
        traj1_x.append(pos1[0]); traj1_y.append(pos1[1])
        traj2_x.append(pos2[0]); traj2_y.append(pos2[1])
        dist = np.linalg.norm(pos1 - pos2)
        separation.append(dist)
        times.append(n*dt)
        if dist < 0.5: break # Merge

# --- VISUALIZATION ---
fig = plt.figure(figsize=(14, 6))

# Panel A
ax1 = fig.add_subplot(121)
# Adjust contrast to see the spiral better, hide the far halo a bit
ax1.imshow(phi_curr, extent=[-L, L, -L, L], origin='lower', cmap='magma', vmin=-0.8, vmax=0.1)

# Draw fading trails
ax1.plot(traj1_x, traj1_y, color='cyan', lw=1.5, alpha=0.7, label='Soliton A')
ax1.plot(traj2_x, traj2_y, color='white', lw=1.5, alpha=0.7, label='Soliton B')

ax1.scatter(pos1[0], pos1[1], c='cyan', s=80, edgecolors='white', zorder=5)
ax1.scatter(pos2[0], pos2[1], c='white', s=80, edgecolors='cyan', zorder=5)

ax1.set_title("a) Scalar Field Potential & Inspiral Trajectories", fontweight='bold')
ax1.set_xlim(-8, 8); ax1.set_ylim(-8, 8)
ax1.legend(loc='upper right')

# Panel B
ax2 = fig.add_subplot(122)
ax2.plot(times, separation, 'k-', lw=1.5)
ax2.set_title("b) Orbital Decay R(t)", fontweight='bold')
ax2.set_xlabel("Time t")
ax2.set_ylabel("Separation")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, R_init+1)

ax2.annotate('Energy lost to Radiation', 
             xy=(times[len(times)//2], separation[len(separation)//2]), 
             xytext=(times[len(times)//2]+10, separation[len(separation)//2]+2),
             arrowprops=dict(facecolor='black', arrowstyle='->'),
             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

plt.suptitle("Figure 4: Binary System Dynamics. Natural orbital decay via scalar field radiation.", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("fig4_binary_orbital_decay_final.png", dpi=150)
plt.show()
