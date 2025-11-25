import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fftpack import fft2, ifft2

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE 
# =============================================================================
# SCRIPT: fig3_relativistic_scattering_soler.py
# PURPOSE: Simulation of Relativistic Soliton Collisions in (3+1)D.
#
# THEORETICAL BACKGROUND:
# Unlike linear wave packets which superimpose and pass through each other,
# Soler solitons are topological excitations stabilized by self-interaction:
# L_NL = -G * (rho / (1 + S*rho)).
#
# In high-energy collisions (v ~ 0.6c), these extended particles exhibit 
# complex non-linear dynamics. Depending on the relative phase and impact 
# parameter, they can merge into an excited "breather", reflect, or pass 
# through with a phase shift (quasi-elastic scattering).
#
# NUMERICAL METHOD:
# We employ the Split-Step Fourier Method (SSFM) on a 2D slice (Y-Z plane).
# This method is symplectic and unitary, preserving the global charge Q
# to machine precision, which is crucial for distinguishing physical
# scattering outcomes from numerical dissipation.
# =============================================================================

# --- CONFIGURATION ---
# Grid parameters (High resolution for topological stability)
Ly, Lz = 30.0, 30.0  # Physical domain size
Ny, Nz = 256, 256    # Grid resolution
dt = 0.015           # Time step
T_max = 22.0         # Simulation duration
m = 1.0              # Rest mass

# Soler Nonlinearity Parameters (Matched to Preprint Fig. 3)
G_soler = 2.0
S_soler = 3.0

# Collision Kinematics
p_momentum = 0.8     # Initial momentum (approx v=0.6c)
z_offset = 7.0       # Initial half-separation
rel_phase = 0.0      # Relative phase between spinors

# --- DIRAC MATRICES & PHYSICS UTILS ---
# Standard representation in 4D
gamma0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=complex)
gamma2 = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=complex)
gamma3 = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=complex)

alpha_y = np.dot(gamma0, gamma2)
alpha_z = np.dot(gamma0, gamma3)
beta    = gamma0

def gaussian_packet(Y, Z, y0, z0, py, pz, width=1.4, phase_offset=0.0):
    """Generates a boosted spinor wave packet."""
    r2 = (Y - y0)**2 + (Z - z0)**2
    envelope = np.exp(-r2 / (2 * width**2))
    phase = np.exp(1j * (py * Y + pz * Z + phase_offset))
    
    # Spin-up state initialization
    u = np.array([1, 0, 0, 0], dtype=complex) 
    psi = np.zeros((4, Y.shape[0], Y.shape[1]), dtype=complex)
    for i in range(4):
        psi[i, :, :] = u[i] * envelope * phase
    return psi

# --- SPECTRAL SOLVER SETUP ---
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(-Lz/2, Lz/2, Nz)
Y, Z = np.meshgrid(y, z, indexing='ij')

# Frequency domain
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
kz = 2 * np.pi * np.fft.fftfreq(Nz, d=Lz/Nz)
KY, KZ = np.meshgrid(ky, kz, indexing='ij')

# Kinetic Propagator (Exact integration in k-space)
E_k = np.sqrt(KY**2 + KZ**2 + m**2)
cos_term = np.cos(E_k * dt)
sin_term = np.sin(E_k * dt) / (E_k + 1e-12)

# Pre-allocate tensors
Identity = np.eye(4, dtype=complex)[:, :, np.newaxis, np.newaxis]
H_k_grid = (np.einsum('ij,kl->ijkl', alpha_y, KY) + 
            np.einsum('ij,kl->ijkl', alpha_z, KZ) + 
            np.einsum('ij,kl->ijkl', beta, np.ones_like(KY)*m))
U_kinetic = cos_term * Identity - 1j * sin_term * H_k_grid

def step_evolution(psi):
    """Performs one SSFM time step: V(dt/2) -> K(dt) -> V(dt/2)"""
    
    # 1. Nonlinear Half-Step
    # Calculate scalar density for Soler potential
    rho_s = np.abs(np.abs(psi[0])**2 + np.abs(psi[1])**2 - np.abs(psi[2])**2 - np.abs(psi[3])**2)
    V_pot = - G_soler * rho_s / (1.0 + S_soler * rho_s)
    psi = psi * np.exp(-1j * V_pot * dt/2)
    
    # 2. Kinetic Step (FFT)
    psi_k = np.zeros_like(psi)
    for i in range(4): psi_k[i] = fft2(psi[i])
    
    # Matrix multiplication in k-space
    psi_k = np.einsum('ij...,j...->i...', U_kinetic, psi_k)
    
    for i in range(4): psi[i] = ifft2(psi_k[i])
    
    # 3. Nonlinear Half-Step
    rho_s = np.abs(np.abs(psi[0])**2 + np.abs(psi[1])**2 - np.abs(psi[2])**2 - np.abs(psi[3])**2)
    V_pot = - G_soler * rho_s / (1.0 + S_soler * rho_s)
    psi = psi * np.exp(-1j * V_pot * dt/2)
    
    return psi

# --- SIMULATION EXECUTION ---
print(f"Initializing Collision... G={G_soler}, Momentum={p_momentum}")

# Create two counter-propagating solitons
psi1 = gaussian_packet(Y, Z, 0, -z_offset, 0, p_momentum, phase_offset=0)
psi2 = gaussian_packet(Y, Z, 0, +z_offset, 0, -p_momentum, phase_offset=rel_phase) 
psi = (psi1 + psi2) * 3.0 # Amplitude scaling to trigger nonlinearity

# Storage
snapshots = {}
max_rho_history = []
steps = int(T_max / dt)
collision_detected = False

print(f"Running Evolution ({steps} steps)...")

for n in range(steps):
    psi = step_evolution(psi)
    t = n * dt
    
    rho = np.sum(np.abs(psi)**2, axis=0)
    current_max = np.max(rho)
    max_rho_history.append(current_max)
    
    # Capture 1: Initial State
    if n == 0:
        snapshots['start'] = rho.copy()
    
    # Capture 2: Collision Peak (Auto-detection)
    # Logic: We look for the moment density starts decreasing after a long rise
    if not collision_detected and t > 5.0 and len(max_rho_history) > 5:
        if current_max < max_rho_history[-2]:
             snapshots['collision'] = rho.copy()
             collision_detected = True
             print(f"Collision Snapshot Captured at t={t:.2f}")

    # Capture 3: Final State
    if n == steps - 1:
        snapshots['final'] = rho.copy()

# --- VISUALIZATION ---
print("Rendering Figure 3...")
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.1]) 

# Plot 1: Approach
ax1 = plt.subplot(gs[0])
im1 = ax1.imshow(snapshots['start'], extent=[-Lz/2, Lz/2, -Ly/2, Ly/2], 
           origin='lower', cmap='inferno', vmax=np.max(snapshots['start'])*0.8)
ax1.set_title("Initial State (Approach)", fontweight='bold', fontsize=11)
ax1.set_xlabel("Z axis")
ax1.set_ylabel("Y axis")

# Plot 2: Collision
ax2 = plt.subplot(gs[1])
# Use dynamic range from collision for best contrast
im2 = ax2.imshow(snapshots['collision'], extent=[-Lz/2, Lz/2, -Ly/2, Ly/2], 
           origin='lower', cmap='inferno')
ax2.set_title("Collision (Merger/Max Density)", fontweight='bold', fontsize=11)
ax2.set_xlabel("Z axis")
ax2.set_yticklabels([])

# Plot 3: Scattering
ax3 = plt.subplot(gs[2])
im3 = ax3.imshow(snapshots['final'], extent=[-Lz/2, Lz/2, -Ly/2, Ly/2], 
           origin='lower', cmap='inferno') 
ax3.set_title("Final State (Scattering/Bound)", fontweight='bold', fontsize=11)
ax3.set_xlabel("Z axis")
ax3.set_yticklabels([])

# Colorbar
cbar = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("Spinor Density", fontsize=9)

plt.suptitle(f"(3+1)D Relativistic Head-on Collision\nParameters: G={G_soler}, Soler={S_soler}, Momentum p={p_momentum}", 
             y=0.98, fontsize=13)

plt.tight_layout()
plt.savefig("fig3_relativistic_scattering_soler.png", dpi=150, bbox_inches='tight')
print("Figure saved successfully.")
plt.show()
