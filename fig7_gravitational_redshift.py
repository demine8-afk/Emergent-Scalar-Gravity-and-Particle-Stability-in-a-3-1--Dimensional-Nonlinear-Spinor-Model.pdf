import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CODE METADATA & PHYSICS ESSENCE
# =============================================================================
# SCRIPT: fig7_gravitational_redshift.py
# PURPOSE: Quantitative demonstration of Gravitational Redshift.
#
# THEORETICAL BACKGROUND:
# A soliton (breather) is a localized oscillating solution of the field equations.
# Its oscillation frequency omega corresponds to its total energy E = h_bar * omega.
# In the rest frame, E ~ m (mass).
#
# When the soliton is placed inside an attractive scalar potential well (Phi < 0),
# its total energy is lowered due to the binding energy:
# E_probe approx m + <Phi>.
#
# Consequently, the oscillation frequency decreases relative to vacuum (Redshift).
#
# SIMULATION:
# We simulate the time evolution of a nonlinear wave packet in two scenarios:
# 1. Vacuum (Phi = 0) -> Reference Frequency.
# 2. Potential Well (Phi = -0.3) -> Redshifted Probe Frequency.
# We track the instantaneous phase velocity d(phase)/dt.
# =============================================================================

# --- CONFIGURATION ---
L = 30.0                # Domain Size
N = 128                 # Grid Resolution
dt = 0.005              # Smaller time step for smooth freq tracking
steps = 4000            # Duration

# Physics Constants
mass = 1.0              # Rest mass (shifts frequency to ~1.0)
Phi_depth = -0.3        # Depth of the gravitational well
R_well = 8.0            # Wide well
Nonlinearity = -1.0     # Self-focusing coefficient

# Grid
x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)
R2 = X**2 + Y**2

# --- SIMULATION FUNCTION ---
def run_simulation(potential_depth):
    """
    Evolves a Gaussian packet in a given potential and tracks frequency.
    """
    # 1. Potential Field
    V_ext = potential_depth * np.exp(-R2 / (R_well**2))
    
    # 2. Initial State (Gaussian)
    # We set width=2.5 which is slightly off-equilibrium to induce "breathing"
    # This reproduces the oscillation seen in the preprint.
    width = 2.5
    Psi = np.exp(-R2 / (2 * width**2)) + 0j
    Psi /= np.sqrt(np.sum(np.abs(Psi)**2) * (L/N)**2) # Normalize
    
    # Precompute FFT operators
    kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2
    
    # Split-Step Operators
    # Kinetic: exp(-i * k^2/2m * dt)
    # Note: We treat the mass term in the phase manually
    Op_K = np.exp(-1j * (K2 / (2.0 * mass)) * dt)
    
    # Monitoring arrays
    time_axis = []
    freq_axis = []
    
    # Initial Phase
    prev_phase = np.angle(Psi[N//2, N//2])
    
    # Loop
    for t in range(steps):
        # A. Potential Step (External + Nonlinear + Rest Mass)
        # The term 'mass' is added to V to simulate Rest Energy E ~ m
        Density = np.abs(Psi)**2
        V_total = V_ext + Nonlinearity * Density + mass
        
        Op_V = np.exp(-1j * V_total * dt)
        Psi = Psi * Op_V
        
        # B. Kinetic Step
        Psi_k = np.fft.fft2(Psi)
        Psi_k = Psi_k * Op_K
        Psi = np.fft.ifft2(Psi_k)
        
        # C. Measurement (every 10 steps to reduce noise)
        if t % 10 == 0:
            current_phase = np.angle(Psi[N//2, N//2])
            
            # Unwrap phase jump
            d_phase = current_phase - prev_phase
            if d_phase > np.pi: d_phase -= 2*np.pi
            if d_phase < -np.pi: d_phase += 2*np.pi
            
            # Instantaneous Frequency omega = -d(phase)/dt
            # Since we skip 10 steps, dt_effective = 10*dt
            inst_freq = -d_phase / (10 * dt)
            
            time_axis.append(t * dt)
            freq_axis.append(inst_freq)
            
            prev_phase = current_phase
        
    return np.array(time_axis), np.array(freq_axis), Density, V_ext

# --- EXECUTE SIMULATIONS ---
print("Running Vacuum Simulation...")
t_vac, w_vac, rho_vac, v_vac = run_simulation(potential_depth=0.0)

print("Running Gravity Well Simulation...")
t_probe, w_probe, rho_probe, v_probe = run_simulation(potential_depth=Phi_depth)

# Calculate Mean Frequencies
# We take the average over the stable part
mean_w_vac = np.mean(w_vac[50:]) 
mean_w_probe = np.mean(w_probe[50:])
shift = mean_w_probe - mean_w_vac

print(f"Result: Vac={mean_w_vac:.4f}, Probe={mean_w_probe:.4f}, Shift={shift:.4f}")

# --- VISUALIZATION ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(wspace=0.2)

# PANEL 1: SETUP VISUALIZATION
im1 = ax1.imshow(rho_probe, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='magma')

# Overlay Potential Well contours
levels = [-Phi_depth * 0.2, -Phi_depth * 0.5, -Phi_depth * 0.8] 
ax1.contour(X, Y, -v_probe, levels=levels, colors='cyan', linestyles='--', linewidths=1.2)

ax1.set_title("Setup: Reference vs Probe", fontweight='bold', fontsize=11)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_facecolor('black')

# PANEL 2: FREQUENCY SHIFT
# Plot the oscillating frequencies
# Using rf-strings to fix SyntaxWarning
ax2.plot(t_vac, w_vac, color='blue', label=rf'Reference $\omega \approx {mean_w_vac:.4f}$', linewidth=1.5)
ax2.plot(t_probe, w_probe, color='red', linestyle='--', label=rf'Probe $\omega \approx {mean_w_probe:.4f}$', linewidth=1.8)

ax2.set_title("Emergent Gravitational Redshift (Eigenfrequency Shift)", fontweight='bold', fontsize=11)
ax2.set_xlabel("Time t")
ax2.set_ylabel(r"Frequency $\omega(t)$")
ax2.grid(True, alpha=0.2, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.9)
# Set limits to focus on the data
ax2.set_ylim(min(w_probe)*0.9, max(w_vac)*1.1)

# Text Box with Physics Results
textstr = '\n'.join((
    rf'$\mathbf{{Measured\ Shift}}\ \Delta\omega = {shift:.4f}$',
    rf'Potential Depth $\Phi = {Phi_depth}$',
    r'Result: $E_{gnd} < E_{vac}$'
    ))

props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
ax2.text(0.05, 0.15, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# --- GLOBAL TITLE ---
plt.suptitle(f"Figure 7: Gravitational Redshift. Frequency shift of a probe soliton inside a potential well.", 
             fontsize=13, fontweight='bold', y=0.98)

filename = "fig7_gravitational_redshift.png"
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"Figure saved to {filename}")
plt.show()
