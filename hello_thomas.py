import gdsfactory as gf
import gplugins.gmeep as gm
import matplotlib.pyplot as plt
import numpy as np

# Activate the generic PDK (220 nm Si)
gf.gpdk.PDK.activate()

# Create the TE grating coupler (optimized for low loss)
gc = gf.components.grating_coupler_elliptical(
    fiber_angle=8.0,
    cross_section="strip",
    taper_length=40,
    grating_line_width=0.35,
    n_periods=40,
)

gc.plot()
gc.write_gds("grating_coupler_te_8deg.gds")
print("✅ GDS saved — open in KLayout and press F to fit")

# Run Meep FDTD simulation (fully local, free)
print("\n🚀 Running Meep FDTD simulation (2–8 minutes depending on your CPU)...")
sp = gm.write_sparameters_grating(
    component=gc,
    port_waveguide_name="o1",
    fiber_angle=8.0,
    run=True,
    verbose=True,
)

# Extract and print results
wavelengths = sp["wavelengths"] * 1e6
transmission = 10 * np.log10(np.abs(sp["o2@0,o1@0"]) ** 2)   # coupling loss in dB

min_loss_db = transmission.min()
best_wl = wavelengths[transmission.argmin()]

print(f"\n✅ RESULTS (Meep)")
print(f"Minimum coupling loss: {min_loss_db:.2f} dB at {best_wl:.3f} µm")

# Plot for your application
plt.figure(figsize=(8, 5))
plt.plot(wavelengths, transmission, "b-", linewidth=2)
plt.axvline(1.55, color="red", linestyle="--", label="1550 nm")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Coupling loss (dB)")
plt.title("TE Fiber Grating Coupler – 8° (220 nm Si) — Meep")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Save results
np.savez("grating_coupler_sparameters_meep.npz", **sp)
print("✅ S-parameters saved! Screenshot the plot + GDS for your screener.")
