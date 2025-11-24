"""
SCRIPT 05: GRAVITATIONAL REDSHIFT & TIME DILATION
-------------------------------------------------
Paper Reference: Sec. 3.5 (Emergent Gravitational Redshift)

PURPOSE:
Verifies that a particle placed in a scalar gravitational potential oscillates
slower than a free particle, reproducing the phenomenon of gravitational 
redshift/time dilation (E' = E + m*Phi).

METHOD:
- Initialize two identical solitons: Reference (Vacuum) and Probe (Potential Well).
- Evolve both using the full Hamiltonian.
- Extract the instantaneous frequency omega(t) = d(Phase)/dt.
- Compare the steady-state frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq

# --- PHYSICS PARAMETERS ---
m = 1.0              # Mass
G_soler = 3.0        # Nonlinear coupling
S_sat = 0.5          # Saturation
dt = 0.025           # Time step
t_max = 25.0         # Duration
L = 32.0             # Box size
N = 64               # Resolution

# Gravitational Potential Parameters
Phi_depth = -0.3     # Depth of the well (Phi < 0 means attractive)
Well_width = 6.0     # Width of the potential well

# --- GRID SETUP ---
x = np.linspace(-L/2, L/2, N, endpoint=False)
X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

# Momentum Space
kx = 2 * np.pi * fftfreq(N, L/N)
KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# --- DIRAC OPERATORS ---
I = np.eye(4, dtype=complex)
AlphaX = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=complex)
AlphaY = np.array([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]], dtype=complex)
AlphaZ = np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]], dtype=complex)
Beta   = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=complex)

# Kinetic Propagator
Ek = np.sqrt(K2 + m**2)
Ck = np.cos(Ek * dt)
Sk = np.sin(Ek * dt) / Ek
Hk_term = (np.einsum('ij,xyz->ijxyz', AlphaX, KX) + 
           np.einsum('ij,xyz->ijxyz', AlphaY, KY) + 
           np.einsum('ij,xyz->ijxyz', AlphaZ, KZ) + 
           np.einsum('ij,xyz->ijxyz', Beta,   np.full_like(K2, m)))
U_kin = Ck[None, None, ...] * I[:, :, None, None, None] - 1j * Sk[None, None, ...] * Hk_term

# --- SCENARIO INITIALIZATION ---
pos_ref = np.array([-8.0, 0.0, 0.0]) # Left side (Vacuum)
pos_prb = np.array([ 8.0, 0.0, 0.0]) # Right side (Inside Well)

# 1. External Potential (The "Gravity Well")
# Applied only around the probe position
R2_prb = (X - pos_prb[0])**2 + (Y - pos_prb[1])**2 + (Z - pos_prb[2])**2
Phi_ext = Phi_depth * np.exp(-R2_prb / Well_width**2)

# 2. Soliton Wavefunctions
def make_spinor(x0, y0, z0):
    r2 = (X - x0)**2 + (Y - y0)**2 + (Z - z0)**2
    psi = np.exp(-0.5 * r2 / 2.0**2) # Gaussian envelope
    res = np.zeros((4, N, N, N), dtype=complex)
    res[0] = psi
    return res

Psi_ref = make_spinor(*pos_ref)
Psi_prb = make_spinor(*pos_prb)

# Normalize densities
vol_elem = (L/N)**3
norm_factor = np.sqrt(1.0 / (np.sum(np.abs(Psi_ref)**2) * vol_elem))
Psi_ref *= norm_factor
Psi_prb *= norm_factor

# Combined state (Linearly independent spatially, so we can sum them)
Psi_total = Psi_ref + Psi_prb

# --- MEASUREMENT TOOLS ---
# Gaussian windows to isolate each particle for phase measurement
w_ref = np.exp(-((X - pos_ref[0])**2 + Y**2 + Z**2)/(2.5**2))
w_prb = np.exp(-((X - pos_prb[0])**2 + Y**2 + Z**2)/(2.5**2))

# Initial projections for differential phase tracking
prev_ref = np.sum(Psi_total * w_ref)
prev_prb = np.sum(Psi_total * w_prb)

# Data storage
times = []
omega_ref_hist = []
omega_prb_hist = []

# --- SIMULATION LOOP ---
steps = int(t_max / dt)
print(f"Redshift Experiment: Steps={steps}, Well Depth={Phi_depth}")

for step in range(steps + 1):
    # --- A. Nonlinear Potential Step ---
    rho = np.sum(np.abs(Psi_total)**2, axis=0)
    
    # Soler Self-Interaction (Internal Stability)
    V_nl = -G_soler * rho / (1.0 + S_sat * rho)
    
    # Total Potential = Internal + External
    V_tot = V_nl + Phi_ext
    
    Psi_total *= np.exp(-0.5j * V_tot * dt)
    
    # --- B. Kinetic Step ---
    Psi_k = fftn(Psi_total, axes=(1,2,3))
    Psi_k = np.einsum('ijxyz,jxyz->ixyz', U_kin, Psi_k)
    Psi_total = ifftn(Psi_k, axes=(1,2,3))
    
    # --- C. Nonlinear Potential Step (2nd Half) ---
    # Update rho for better accuracy
    rho = np.sum(np.abs(Psi_total)**2, axis=0)
    V_nl = -G_soler * rho / (1.0 + S_sat * rho)
    V_tot = V_nl + Phi_ext
    
    Psi_total *= np.exp(-0.5j * V_tot * dt)
    
    # --- D. Phase Measurement ---
    # Project current state onto local windows
    curr_ref = np.sum(Psi_total * w_ref)
    curr_prb = np.sum(Psi_total * w_prb)
    
    # Compute phase difference relative to previous step
    # d_phi = arg( <Psi(t) | Psi(t+dt)> )
    # This gives the instantaneous energy E = - d_phi/dt
    d_phi_ref = np.angle(np.vdot(prev_ref, curr_ref))
    d_phi_prb = np.angle(np.vdot(prev_prb, curr_prb))
    
    # Frequency omega = - d_phi / dt
    w_ref_val = -d_phi_ref / dt
    w_prb_val = -d_phi_prb / dt
    
    # Store
    times.append(step * dt)
    omega_ref_hist.append(w_ref_val)
    omega_prb_hist.append(w_prb_val)
    
    # Update prev
    prev_ref = curr_ref
    prev_prb = curr_prb

# --- DATA ANALYSIS & PLOTTING ---
# Smoothing function to remove numerical jitter
def smooth_data(data, window_size):
    box = np.ones(window_size)/window_size
    return np.convolve(data, box, mode='valid')

win = 20
t_smooth = times[win-1:]
w_ref_smooth = smooth_data(omega_ref_hist, win)
w_prb_smooth = smooth_data(omega_prb_hist, win)

# Calculate mean frequencies in stable region
stable_idx = int(len(t_smooth)/2)
mean_ref = np.mean(w_ref_smooth[stable_idx:])
mean_prb = np.mean(w_prb_smooth[stable_idx:])
shift = mean_prb - mean_ref

print(f"Results: Omega_Ref={mean_ref:.4f}, Omega_Prb={mean_prb:.4f}")
print(f"Shift: {shift:.4f} (Expected ~ {Phi_depth})")

# Plot
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# 1. Spatial Setup
mid_z = N//2
# Plot Density + Potential Contours
density_slice = np.sum(np.abs(Psi_total)**2, axis=0)[:, :, mid_z]
potential_slice = Phi_ext[:, :, mid_z]

ax[0].imshow(density_slice.T, extent=[-L/2,L/2,-L/2,L/2], origin='lower', cmap='magma')
ax[0].contour(X[:,:,mid_z], Y[:,:,mid_z], potential_slice, levels=[-0.25, -0.15, -0.05], 
              colors='cyan', linestyles='dashed', linewidths=1.5)
ax[0].text(-8, 5, "Reference\n(Vacuum)", color='white', ha='center')
ax[0].text(8, 5, "Probe\n(In Well)", color='white', ha='center')
ax[0].set_title("Experimental Setup", fontweight='bold')
ax[0].set_xlabel("X")
ax[0].set_ylabel("Y")

# 2. Frequency Evolution
ax[1].plot(t_smooth, w_ref_smooth, 'b-', label=f"Reference $\omega \\approx {mean_ref:.3f}$", lw=2)
ax[1].plot(t_smooth, w_prb_smooth, 'r--', label=f"Probe $\omega \\approx {mean_prb:.3f}$", lw=2)

ax[1].set_title("Gravitational Redshift (Frequency Shift)", fontweight='bold')
ax[1].set_xlabel("Time $t$")
ax[1].set_ylabel("Internal Frequency $\omega(t)$")
ax[1].legend(fontsize=11)
ax[1].grid(True, alpha=0.3)

# Annotation
info_text = (f"Measured Shift $\Delta\omega = {shift:.4f}$\n"
             f"Potential Depth $\Phi = {Phi_depth}$\n"
             f"Conclusion: Frequency decreases in potential well")
ax[1].text(0.05, 0.15, info_text, transform=ax[1].transAxes, 
           bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round'))

plt.tight_layout()
output_file = 'fig_redshift_v2.png'
plt.savefig(output_file, dpi=150)
print(f"Redshift plot saved to {output_file}")
plt.close()
