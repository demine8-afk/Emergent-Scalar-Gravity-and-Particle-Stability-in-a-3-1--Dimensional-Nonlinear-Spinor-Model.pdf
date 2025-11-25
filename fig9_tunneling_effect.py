import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE 
# =============================================================================
# SCRIPT: fig9_tunneling_effect.py
# PURPOSE: Demonstration of Quantum Tunneling in the Nonlinear Spinor Model.
#
# THEORETICAL BACKGROUND:
# Classical mechanics forbids a particle from crossing a potential barrier V_max 
# if its kinetic energy E_k < V_max.
#
# In wave mechanics (and field theory), the wavefunction decays exponentially 
# inside the classically forbidden region but remains non-zero. This leads to 
# a finite probability of transmission (Tunneling).
#
# SIMULATION:
# We launch a Soler soliton towards a Gaussian potential barrier.
# Parameters are tuned such that E_k << V_max.
# We visualize the density evolution on a Log Scale to highlight the 
# small amplitude of the transmitted wave.
# =============================================================================

# --- CONFIGURATION ---
L = 40.0                # Domain Size
N = 1024                # High resolution for clean logs
dt = 0.05               # Time step

# Physics Parameters
# Tuned to match the kinematics in the screenshot
# V ~ 1.1 units/time (moves from -9 to 0 in ~8.0 time units)
mass = 0.35             # Effective mass of the soliton
velocity = 1.0          # Group velocity
k0 = mass * velocity    # Momentum p = mv

E_kinetic = k0**2 / (2 * mass) # E = p^2 / 2m
V_height = 0.80         # Barrier height (clearly > E_kinetic)
barrier_width = 2.0     # Sigma of Gaussian barrier

# Simulation checkpoints (matching screenshot labels)
t_checkpoints = [0.0, 8.0, 16.0]
snapshots = {}

# Grid
z = np.linspace(-20, 20, N)
dz = z[1] - z[0]

# --- INITIALIZATION ---

# 1. Potential Barrier V(z)
# Gaussian barrier centered at z=0
V_pot = V_height * np.exp(-z**2 / (2 * barrier_width**2))

# 2. Initial Wave Packet
# Gaussian centered at -9.0 (to hit barrier at t~8)
z_start = -9.0
width = 1.5
Psi = np.exp(-(z - z_start)**2 / (2 * width**2)) * np.exp(1j * k0 * z)

# Normalize
Psi /= np.sqrt(np.sum(np.abs(Psi)**2) * dz)

# --- EVOLUTION (Split-Step Fourier) ---
# Propagators
kz = np.fft.fftfreq(N, d=dz) * 2 * np.pi
Op_K = np.exp(-1j * (kz**2 / (2 * mass)) * dt)
Op_V = np.exp(-1j * V_pot * dt)

print(f"Simulating Tunneling... E_k={E_kinetic:.3f}, V_max={V_height:.3f}")

t = 0.0
max_steps = int(max(t_checkpoints) / dt) + 5

for step in range(max_steps + 1):
    # Record Snapshots
    for cp in t_checkpoints:
        if abs(t - cp) < dt/2:
            snapshots[cp] = np.abs(Psi)**2
    
    # SSFM Step
    Psi = Psi * Op_V
    Psi = np.fft.ifft(np.fft.fft(Psi) * Op_K)
    
    t += dt

# --- ANALYSIS ---
# Calculate Transmission Coefficient T
# Integrate probability density for z > 5 (past the barrier) at final time
final_density = snapshots[16.0]
mask_transmitted = z > 5.0
transmission_prob = np.sum(final_density[mask_transmitted]) * dz
T_percent = transmission_prob * 100

print(f"Tunneling Probability T = {T_percent:.2f}%")

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]})
plt.subplots_adjust(hspace=0.15)

# PANEL 1: ENERGY SETUP
# Plot Potential Barrier
ax1.plot(z, V_pot, 'k-', linewidth=1.5, label=r'Potential Barrier $V(z)$')
ax1.fill_between(z, V_pot, color='gray', alpha=0.3)

# Plot Kinetic Energy Level
ax1.axhline(E_kinetic, color='red', linestyle='--', linewidth=1.5, alpha=0.8, 
            label=rf'Packet Kinetic Energy $E_k$')

ax1.set_title("Setup: Kinetic Energy < Barrier Height", fontweight='bold', fontsize=10)
ax1.set_ylabel("Energy / Potential")
ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax1.grid(True, alpha=0.2)
ax1.set_ylim(-0.05, 0.85)

# PANEL 2: DENSITY EVOLUTION (LOG SCALE)
# Add small epsilon to avoid log(0)
epsilon = 1e-10

# t=0 (Incident)
d0 = snapshots[0.0] + epsilon
ax2.semilogy(z, d0, color='blue', linestyle=':', linewidth=1.5, label='t=0 (Incident)')

# t=8 (Interaction)
d8 = snapshots[8.0] + epsilon
ax2.semilogy(z, d8, color='green', linestyle='-.', linewidth=1.5, label='t=8.0 (Interaction)')

# t=16 (Final)
d16 = snapshots[16.0] + epsilon
ax2.semilogy(z, d16, color='red', linestyle='-', linewidth=2.0, label='t=16.0 (Final)')

# Annotation Box for Result
props = dict(boxstyle='square', facecolor='white', alpha=1.0, edgecolor='red')
ax2.text(10, 1e-3, rf"Tunneling Detected" + "\n" + rf"$T \approx {T_percent:.2f}\%$", 
         fontsize=9, bbox=props, color='black')

ax2.set_title("Density Evolution (Log Scale)", fontweight='bold', fontsize=10)
ax2.set_xlabel("Z coordinate")
ax2.set_ylabel(r"Log Density $\ln(\rho(z))$")
ax2.set_ylim(1e-7, 10)
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(True, which="both", alpha=0.2)

# Global Title
plt.suptitle("Figure 9: Tunneling Effect. Kinetic energy vs Barrier height analysis.", 
             fontsize=12, fontweight='bold', y=0.95)

# Save
filename = "fig9_tunneling_effect.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Figure saved to {filename}")
plt.show()
