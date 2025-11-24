"""
SCRIPT 04: GRAVITATIONAL LENSING (3D)
-------------------------------------
Paper Reference: Sec. 3.8 (Gravitational Lensing)

PURPOSE:
Demonstrates that the scalar potential acts as a refractive medium for matter waves.
A broad Gaussian beam is fired past a massive central object. The attractive 
potential causes the phase velocity to decrease (v_phase ~ 1/n), bending wavefronts 
inward and creating a focal point.

METHOD:
- Full (3+1)D Dirac evolution.
- Static Gaussian Potential Well (Lens).
- Analysis of energy density convergence in the far field.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fftn, ifftn, fftfreq

# --- PHYSICS PARAMETERS ---
m = 0.5              # Light mass to enhance diffraction effects
dt = 0.05            # Time step
t_max = 20.0         # Duration to propagate past the focal point
L = 40.0             # Domain size
N = 64               # Resolution (64^3)

# Lens Properties
Lens_Strength = -1.5 # Negative = Attractive = Convex Lens
Lens_Width = 5.0     
Lens_Pos = np.array([0.0, 0.0, 0.0])

# Beam Properties
k_beam = 1.5         # Longitudinal momentum
Beam_Width = 12.0    # Wide beam to clearly show focusing

# --- GRID SETUP ---
# Cubic grid
x = np.linspace(-L/2, L/2, N, endpoint=False)
y = x
z = x # Explicitly define z for later masking
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Momentum Grid
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

# 1. Gravitational Lens Potential (Static)
# Gaussian well: Phi(r) = Phi_0 * exp(-r^2 / sigma^2)
R2 = (X - Lens_Pos[0])**2 + (Y - Lens_Pos[1])**2 + (Z - Lens_Pos[2])**2
Phi_Lens = Lens_Strength * np.exp(-R2 / Lens_Width**2)

# 2. Initial Beam
# We construct a wide Gaussian packet moving in +Z direction
# Start at Z = -15
psi_spatial = np.zeros_like(X, dtype=np.complex128)
# Spatial Envelope: Wide in X/Y, narrow in Z (packet)
psi_spatial = np.exp(-0.5 * ((Z + 15.0)**2)/2.0**2) * np.exp(-0.5 * (X**2 + Y**2)/Beam_Width**2)

# Momentum Boost: multiply by plane wave
psi_spatial *= np.exp(1j * k_beam * Z)

# Embed in Spinor (Spin-up)
Psi = np.zeros((4, N, N, N), dtype=complex)
Psi[0] = psi_spatial
norm = np.sum(np.abs(Psi)**2)
Psi /= np.sqrt(norm)

# --- SIMULATION LOOP ---
steps = int(t_max / dt)
print(f"Lensing Simulation: Beam k={k_beam}, Lens Strength={Lens_Strength}")

for step in range(steps + 1):
    # Split-Step: Potential -> Kinetic -> Potential
    
    # Total Potential (Just the lens here, neglecting self-gravity for linear optics demo)
    V_tot = Phi_Lens 
    
    # Half-step V
    Psi *= np.exp(-0.5j * V_tot * dt)
    
    # Full-step T (FFT)
    Psi_k = fftn(Psi, axes=(1,2,3))
    Psi_k = np.einsum('ijxyz,jxyz->ixyz', U_kin, Psi_k)
    Psi = ifftn(Psi_k, axes=(1,2,3))
    
    # Half-step V
    Psi *= np.exp(-0.5j * V_tot * dt)
    
    if step % 50 == 0:
        print(f"Step {step}/{steps}")

# --- ANALYSIS & PLOTTING ---

# 1. Extract Slice for Visualization
# We look at the final state energy density in the X-Z plane (at Y=0)
rho_final = np.sum(np.abs(Psi)**2, axis=0)
xz_slice = rho_final[:, N//2, :]

# 2. Plotting
fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# Heatmap of Energy Density
# Transpose slice so Z is vertical (Propagation up) or horizontal.
# Here we set Z on Y-axis (vertical), X on X-axis.
im = ax.imshow(xz_slice.T, extent=[-L/2, L/2, -L/2, L/2], origin='lower', 
               cmap='inferno', vmax=np.max(xz_slice)*0.8, interpolation='bicubic')
cbar = fig.colorbar(im, ax=ax, pad=0.02)
cbar.set_label("Beam Energy Density")

# Overlay Lens Contours
X_slice = X[:, N//2, :]
Z_slice = Z[:, N//2, :]
Phi_slice = Phi_Lens[:, N//2, :]
ax.contour(X_slice, Z_slice, Phi_slice, levels=[-1.0, -0.5, -0.2], 
           colors='cyan', linestyles='dashed', linewidths=1, alpha=0.7)
ax.text(2, -2, "Lens\nPotential", color='cyan', fontsize=8)

# Decorate Axes
ax.set_title(f"Gravitational Lensing Simulation\nScalar Potential $\Phi(0)={Lens_Strength}$", 
             fontweight='bold', fontsize=12)
ax.set_xlabel("Transverse Position X [Compton lengths]")
ax.set_ylabel("Propagation Direction Z [Compton lengths]")
ax.axvline(0, color='white', linestyle=':', alpha=0.2) # Optical axis

# 3. Automatic Focal Point Detection
# We look for the maximum intensity in the region past the lens (Z > 5)
mask_region = (z > 5.0)
sub_slice = xz_slice[:, mask_region] # Slice the data array

if sub_slice.size > 0:
    # Find max index in the sub-slice
    max_idx_flat = np.argmax(sub_slice)
    max_idx = np.unravel_index(max_idx_flat, sub_slice.shape)
    
    # Convert local indices back to global coordinates
    fx_idx = max_idx[0]
    # The Z index needs offset by the start of the mask
    z_indices = np.where(mask_region)[0]
    fz_idx = z_indices[max_idx[1]]
    
    fx = x[fx_idx]
    fz = z[fz_idx]
    
    # Mark on plot
    ax.plot(fx, fz, 'wx', markersize=10, markeredgewidth=2)
    ax.text(fx + 1.5, fz, f"FOCAL POINT\nZ={fz:.1f}", color='white', 
            fontweight='bold', fontsize=9, va='center')

plt.tight_layout()
output_filename = 'fig_lensing_test_v2.png'
plt.savefig(output_filename, dpi=150)
print(f"Lensing visualization saved to {output_filename}")
plt.close()
