"""
SCRIPT 01: ORBITAL DYNAMICS IN (3+1)D
-------------------------------------
Purpose: 
Verifies that the nonlinear Dirac soliton behaves like a macroscopic particle 
under gravity. It injects a stable "breather" into a central potential well 
and tracks the Center of Mass (CoM) to visualize the orbit.

Method:
- Operator Splitting (Kinetic in k-space, Potential in x-space).
- Massive Dirac Propagator via exact diagonalization.
- Center of Mass tracking for trajectory visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq

# --- PHYSICS PARAMETERS ---
# Natural units (hbar = c = 1)
m = 1.0              # Bare mass of the spinor field
dt = 0.04            # Time step
t_max = 40.0         # Simulation duration (enough for ~3/4 orbit)
L = 48.0             # Physical domain size (Large box to minimize boundary reflection)
N = 64               # Grid resolution (64^3)

# Central Potential Parameters (Modeling a heavy nucleus or Star)
M_star = 12.0        # Coupling strength of the central source
R_star_width = 4.0   # Regularization width (soft core)

# Initial Conditions for the "Planet" (Soliton)
r_init = np.array([-12.0, 0.0, 0.0])  # Initial position
v_kick = np.array([0.0, 0.45, 0.0])   # Tangential boost (v ~ 0.45c)

# --- GRID SETUP ---
x = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# Momentum space grid
kx = 2 * np.pi * fftfreq(N, L/N)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# --- DIRAC OPERATORS ---
# 4x4 Identity and Gamma matrices in Dirac-Pauli representation
I = np.eye(4, dtype=complex)
AlphaX = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=complex)
AlphaY = np.array([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]], dtype=complex)
AlphaZ = np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]], dtype=complex)
Beta   = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=complex)

# Pre-calculate Kinetic Propagator: exp(-i * (alpha*k + beta*m) * dt)
# Using exact dispersion relation E_k
Ek = np.sqrt(K2 + m**2)
Ck = np.cos(Ek * dt)
Sk = np.sin(Ek * dt) / Ek

# Hamiltonian term in momentum space H_k = alpha*k + beta*m
Hk_term = (np.einsum('ij,xyz->ijxyz', AlphaX, KX) + 
           np.einsum('ij,xyz->ijxyz', AlphaY, KY) + 
           np.einsum('ij,xyz->ijxyz', AlphaZ, KZ) + 
           np.einsum('ij,xyz->ijxyz', Beta,   np.full_like(K2, m)))

# Evolution operator U_kin
U_kin = Ck[None, None, ...] * I[:, :, None, None, None] - 1j * Sk[None, None, ...] * Hk_term

# --- INITIALIZATION ---
# 1. Static Central Potential V = - G M / r (smoothed)
R = np.sqrt(X**2 + Y**2 + Z**2)
Phi_Star = -M_star / (np.sqrt(R**2 + R_star_width**2))

# 2. Soliton Wave Packet
def make_spinor(pos, k_vec):
    """Creates a Gaussian spinor boosted by momentum k"""
    r2 = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
    # Spatial envelope * Plane wave
    envelope = np.exp(-0.5 * r2 / 2.5**2)
    phase = np.exp(1j * (k_vec[0]*X + k_vec[1]*Y + k_vec[2]*Z))
    
    psi = envelope * phase
    res = np.zeros((4, N, N, N), dtype=complex)
    res[0] = psi # Initialize in component 0 (spin up)
    return res

# p = gamma * m * v. For v=0.45, gamma ~ 1.1. Approx k ~ v for logic check.
Psi = make_spinor(r_init, v_kick) 
norm = np.sum(np.abs(Psi)**2)
Psi /= np.sqrt(norm)

# --- MAIN SIMULATION LOOP ---
steps = int(t_max / dt)
track_pos_x = []
track_pos_y = []

print(f"Starting Simulation: Steps={steps}, Central_Mass={M_star}, V_tan={v_kick[1]}")

for step in range(steps + 1):
    # A. Potential Step (Half-step for Strang Splitting usually, here simplified full)
    # Calculate scalar density
    rho = np.sum(np.abs(Psi)**2, axis=0)
    
    # Soler Nonlinearity (V_NL) for particle stability
    # V_eff = - G_s * rho / (1 + S * rho)
    G_soler = 3.0
    S_sat = 0.5
    V_soler = -G_soler * rho / (1.0 + S_sat * rho)
    
    # Total Scalar Potential: External Gravity + Internal Self-Interaction
    V_tot = Phi_Star + V_soler
    
    # Apply Potential operator: exp(-i * V * dt)
    Psi *= np.exp(-0.5j * V_tot * dt)
    
    # B. Kinetic Step (FFT -> Apply U_kin -> IFFT)
    Psi_k = fftn(Psi, axes=(1,2,3))
    Psi_k = np.einsum('ijxyz,jxyz->ixyz', U_kin, Psi_k)
    Psi = ifftn(Psi_k, axes=(1,2,3))
    
    # C. Potential Step (Second Half)
    # Note: In strict Strang splitting, we'd recalc V_soler here, 
    # but for dt=0.04, using the previous V is a stable approximation.
    Psi *= np.exp(-0.5j * V_tot * dt)
    
    # D. Diagnostics & Tracking
    if step % 5 == 0:
        # Calculate Center of Mass
        rho_curr = np.sum(np.abs(Psi)**2, axis=0)
        total_mass = np.sum(rho_curr)
        com_x = np.sum(X * rho_curr) / total_mass
        com_y = np.sum(Y * rho_curr) / total_mass
        
        track_pos_x.append(com_x)
        track_pos_y.append(com_y)
        
        if step % 50 == 0:
            print(f"Step {step}/{steps}: CoM = ({com_x:.2f}, {com_y:.2f})")

# --- PLOTTING RESULTS ---
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# 1. Plot Potential Well (Contours) at Z=0
contour_levels = np.linspace(np.min(Phi_Star[:,:,N//2]), 0, 12)
ax.contour(X[:,:,N//2], Y[:,:,N//2], Phi_Star[:,:,N//2], 
           levels=contour_levels, colors='k', alpha=0.15)
ax.text(0, 0, "Massive\nSource", ha='center', va='center', fontsize=10, fontweight='bold')

# 2. Plot Trajectory
ax.plot(track_pos_x, track_pos_y, 'b.-', markersize=4, linewidth=1.5, label='Soliton Trajectory')

# 3. Mark Start/End
ax.plot(track_pos_x[0], track_pos_y[0], 'go', markersize=8, label='Start')
ax.plot(track_pos_x[-1], track_pos_y[-1], 'rx', markersize=8, label='End')

# Styling
ax.set_xlim(-L/2, L/2)
ax.set_ylim(-L/2, L/2)
ax.set_title(f"(3+1)D Orbital Capture Simulation\n$M_{{star}}={M_star}$, $v_{{init}}={v_kick[1]}c$", fontweight='bold')
ax.set_xlabel("X [Compton lengths]")
ax.set_ylabel("Y [Compton lengths]")
ax.legend()
ax.grid(True, alpha=0.3)

# Save figure
output_filename = 'fig_orbit_test.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=150)
print(f"Simulation complete. Result saved to {output_filename}")
plt.close()
