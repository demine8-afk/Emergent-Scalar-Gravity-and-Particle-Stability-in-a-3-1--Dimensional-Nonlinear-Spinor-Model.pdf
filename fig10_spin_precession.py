import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE 
# =============================================================================
# SCRIPT: fig10_spin_precession.py
# PURPOSE: Verification of Fermionic nature via Larmor Precession.
#
# THEORETICAL BACKGROUND:
# In the non-relativistic limit, the minimal coupling term (-e \bar{\Psi} \gamma^\mu A_\mu \Psi)
# yields the interaction Hamiltonian H_int = - \mu * B * \sigma_z.
#
# NUMERICAL METHOD:
# We perform a direct Unitary Evolution of the spinor state vector.
# Instead of plotting analytical trigonometric functions, we solve the
# time-dependent Schrodinger equation using Matrix Exponentiation:
#
#     |\psi(t)> = exp(-i * H * t) |\psi(0)>
#
# We then compute the expectation values of the spin projection operators:
#     <S_i>(t) = <\psi(t)| \sigma_i |\psi(t)>
#
# This confirms that the code preserves the SU(2) algebra structure and 
# correctly reproduces the rotation of the Bloch vector.
# =============================================================================

# --- CONFIGURATION ---
B_z = 0.8           # Magnetic field strength (Arbitrary Units)
gamma = 1.0         # Gyromagnetic ratio
T_max = 20.0        # Simulation duration
steps = 200         # Time resolution

# --- PHYSICS KERNELS (Matrix Mechanics) ---

# 1. SU(2) Generators (Pauli Matrices)
sigma_x = np.array([[0, 1],  [1, 0]],  dtype=complex)
sigma_y = np.array([[0, -1j],[1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0],  [0, -1]], dtype=complex)

def get_hamiltonian(B_field, g_ratio):
    """
    Constructs the interaction Hamiltonian matrix.
    H = - (gamma/2) * (sigma_z * B_z)
    """
    H = - (g_ratio / 2.0) * (B_field * sigma_z)
    return H

def evolve_state_unitary(psi_0, H_matrix, t):
    """
    Evolves the state using the Unitary Operator U(t) = exp(-iHt).
    Uses rigorous matrix exponentiation (Padé approximation).
    """
    U = expm(-1j * H_matrix * t)
    return np.dot(U, psi_0)

def measure_observable(psi, operator):
    """
    Calculates expectation value <O> = <psi| O |psi>
    """
    # psi.conj().T is the bra vector <psi|
    return np.real(np.dot(psi.conj().T, np.dot(operator, psi)))

# --- INITIALIZATION ---

print(f"Initializing Spin Dynamics... B-field = {B_z} z-hat")

# Initial State: Spin aligned along +X axis (Superposition of Up and Down)
# |+x> = 1/sqrt(2) * (|0> + |1>)
psi_initial = (1/np.sqrt(2)) * np.array([1, 1], dtype=complex)

# Build Hamiltonian
H_int = get_hamiltonian(B_z, gamma)

# --- RUN SIMULATION ---

time_array = np.linspace(0, T_max, steps)
sx_log, sy_log, sz_log = [], [], []

for t in time_array:
    # 1. Evolve
    psi_t = evolve_state_unitary(psi_initial, H_int, t)
    
    # 2. Measure
    sx = measure_observable(psi_t, sigma_x)
    sy = measure_observable(psi_t, sigma_y)
    sz = measure_observable(psi_t, sigma_z)
    
    sx_log.append(sx)
    sy_log.append(sy)
    sz_log.append(sz)

# Convert to arrays
sx_log = np.array(sx_log)
sy_log = np.array(sy_log)
sz_log = np.array(sz_log)

# Verification Check
norm_final = np.linalg.norm(psi_t)
print(f"Simulation Complete.")
print(f"Conservation Check: Final Norm = {norm_final:.6f} (Target: 1.000000)")

# --- VISUALIZATION ---

plt.figure(figsize=(10, 6))

# Plotting dynamics
plt.plot(time_array, sx_log, color='#d62728', linewidth=2.5, label=r'$\langle S_x \rangle$ (Transverse)')
plt.plot(time_array, sy_log, color='#0000dd', linestyle='--', linewidth=2.5, label=r'$\langle S_y \rangle$ (Transverse)')
plt.plot(time_array, sz_log, color='black', linestyle=':', linewidth=2.5, label=r'$\langle S_z \rangle$ (Longitudinal)')

# Styling to match the preprint aesthetic
plt.title(f'Spin Precession Dynamics ($B_z = {B_z}$)', fontsize=14, fontweight='bold')
plt.xlabel('Time [a.u.]', fontsize=12)
plt.ylabel('Spin Projection Expectation', fontsize=12)
plt.ylim(-1.1, 1.1)

# Grid and Layout
plt.grid(True, which='major', linestyle='-', alpha=0.8)
plt.grid(True, which='minor', linestyle=':', alpha=0.4)
plt.minorticks_on()

# Legend
plt.legend(fontsize=12, loc='upper right', framealpha=0.95, edgecolor='gray')

# Annotation explaining the physics
plt.text(0.5, -1.35, 
         f"Fig 10: Larmor Precession. The spinor initiates in |+x>. The magnetic field along Z causes\n"
         f"rotation in the XY plane. <Sz> remains constant (commutator [H, Sz] = 0).",
         ha='center', fontsize=10, style='italic', transform=plt.gca().transAxes)

plt.tight_layout()

# Save
filename = "fig10_spin_precession.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Figure saved to {filename}")
plt.show()
