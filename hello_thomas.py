import gdsfactory as gf

# Activate the generic PDK
gf.gpdk.PDK.activate()

c = gf.Component(name="hello_thomas_chip")

# 1. Create the grating coupler once (uses standard 0.5 µm strip)
gc = gf.components.grating_coupler_elliptical(
    fiber_angle=8.0,
    cross_section="strip"
)

# 2. Left grating coupler
gc_left = c.add_ref(gc)
gc_left.move((0, 0))

# 3. Right grating coupler (mirrored)
gc_right = c.add_ref(gc)
gc_right.mirror()
gc_right.move((800, 0))

# 4. Route a visibly thicker waveguide (fixes the warning + looks like a real chip)
wide_strip = gf.cross_section.strip(width=2.0)   # ← 2 µm wide
gf.routing.route_single(
    c,
    port1=gc_left.ports["o1"],
    port2=gc_right.ports["o1"],
    cross_section=wide_strip
)

# 5. "Hello Thomas" label centered above
text = gf.components.text(
    text="Hello Thomas",
    size=20,
    justify="center",
    layer=(1, 0)
)
text_ref = c.add_ref(text)
text_ref.move((400, 120))

# 6. Die outline (the box that makes it look like a real chip)
die = gf.components.rectangle(
    size=(1200, 600),
    layer=(99, 0)
)
die_ref = c.add_ref(die)
die_ref.move((-200, -300))

# Show and save
c.plot()
c.write_gds("hello_thomas_chip.gds")
print("✅ Done! Open hello_thomas_chip.gds in KLayout")
