import gdsfactory as gf
import gplugins.tidy3d as gt
import matplotlib.pyplot as plt
import numpy as np

# Activate the generic PDK
gf.gpdk.PDK.activate()

# 1. Create the TE grating coupler (8° angle, low-loss elliptical design)
gc = gf.components.grating_coupler_elliptical(
    fiber_angle=8.0,
    cross_section="strip",      # 220 nm Si, width=0.5 µm (meets min 200 nm rule)
    taper_length=30,
    grating_line_width=0.343,
    n_periods=30,
)

gc.plot()
gc.write_gds("grating_coupler_te_8deg.gds")
print("✅ GDS saved — open in KLayout (press F to fit)")

# 2. Run Tidy3D FDTD simulation with debugging
print("\n🚀 Starting Tidy3D simulation (1–3 minutes)...")
try:
    sp = gt.write_sparameters_grating_coupler(
        component=gc,
        port_waveguide_name="o1",      # waveguide port
        fiber_port_prefix="o2",        # fiber port
        run=True,
        verbose=True,
    )

    if sp is None:
        raise ValueError("Simulation returned None (check Tidy3D credits / internet)")

    print("✅ Simulation succeeded! Available S-parameter keys:")
    print(list(sp.keys()))

    # 3. Extract coupling efficiency
    wavelengths = sp["wavelengths"] * 1e6          # µm
    transmission = 10 * np.log10(np.abs(sp["o2@0,o1@0"]) ** 2)   # coupling loss in dB

    min_loss_db = transmission.min()
    best_wl = wavelengths[transmission.argmin()]

    print(f"\n✅ RESULTS")
    print(f"Minimum coupling loss: {min_loss_db:.2f} dB at {best_wl:.3f} µm")

except Exception as e:
    print(f"\n❌ Simulation failed with error: {e}")
    print("   → Try running 'tidy3d configure' again or check your free credits at tidy3d.simulation.cloud")
    raise

# 4. Plot for your submission
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, transmission, "b-", linewidth=2, label="Coupling efficiency")
plt.axvline(1.55, color="red", linestyle="--", label="1550 nm target")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Coupling loss (dB)")
plt.title("TE Fiber Grating Coupler – 8° angle (220 nm Si)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Save results
np.savez("grating_coupler_sparameters.npz", **sp)
print("✅ S-parameters saved!")
