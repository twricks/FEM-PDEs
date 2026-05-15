import gdsfactory as gf

# Activate the generic PDK
gf.gpdk.PDK.activate()

c = gf.Component(name="hello_thomas_chip")

# 1. Create the grating coupler once
gc = gf.components.grating_coupler_elliptical(
    fiber_angle=8.0,
    cross_section="strip"
)

# 2. Add left grating coupler
gc_left = c.add_ref(gc)
gc_left.move((0, 0))

# 3. Add right grating coupler and mirror the INSTANCE (not the cell)
gc_right = c.add_ref(gc)
gc_right.mirror()                    # ← this is the fix
gc_right.move((800, 0))

# 4. Route a waveguide between the two ports
route = gf.routing.route_single(
    gc_left.ports["o1"],
    gc_right.ports["o1"],
    cross_section="strip"
)
c.add_ref(route)

# 5. "Hello Thomas" label centered above
text = gf.components.text(
    text="Hello Thomas",
    size=20,
    justify="center",
    layer=(1, 0)
)
text_ref = c.add_ref(text)
text_ref.move((400, 120))

# 6. Die outline (makes it look like a real chip)
die = gf.components.rectangle(
    size=(1000, 400),
    layer=(99, 0)
)
die_ref = c.add_ref(die)
die_ref.center = (400, 0)

# Show and save
c.plot()
c.write_gds("hello_thomas_chip.gds")
print("✅ Done! Open hello_thomas_chip.gds in KLayout and press F to fit.")
