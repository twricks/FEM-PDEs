##code to model the waves, maximum resonance in a cavity resonator for input tones. 
import numpy as np
from mpi4py import MPI
import gmsh
from dolfinx.io import gmsh as gmshio
from dolfinx import fem, la, mesh, plot
import pyvista
import dolfinx.plot
from dolfinx.fem.petsc import LinearProblem 
from dolfinx.fem import Function, FunctionSpace, Constant
from petsc4py import PETSc
import dolfinx.fem.petsc as petsc
import ufl
import imageio
import os
from dolfinx.plot import vtk_mesh
from dolfinx import default_scalar_type
from petsc4py import PETSc
import dolfinx.fem.petsc as petsc
from pyvista import plotter

import matplotlib as mpl

comm = MPI.COMM_WORLD
rank = comm.rank

R = 1.0 #radius of the cylindrical waveguide
H = 2.0 #height
dv = 0.1 #mesh size

gmsh.initialize() #initializing gmsh
gmsh.option.setNumber("General.Terminal", 1 if rank ==0 else 0) #print terminal messages from the 'leader' process, which is of rank 0

gmsh.model.add("cylindrical_cavity")

cylinder_tag = gmsh.model.occ.addCylinder(0.0,0.0,0.0,0.0,0.0,H,R) #utilizing gmsh's cylinder model; first set of coordinates specifies the bottom center,
#second set the vector from it that defines the axis, and last the radius. 

gmsh.model.occ.synchronize()

#'tagging' the volume
gmsh.model.addPhysicalGroup(3, [cylinder_tag], tag=1) #'physical groups' are gmsh's abstraction to combine elementary geometric entities in meaningful ways
#e.g. for ease of defining math characteristics like boundaries, or physical ones like material. 

gmsh.model.setPhysicalName(3,1,"cylinder volume") #naming the physical group- takes dimensions and a tag

#we tell gmsh to look for entities within a narrow range around where the bottom end of the cylinder is
bottom_cap = gmsh.model.getEntitiesInBoundingBox(
    -R-1e-6,-R-1e-6,-1e-6,R+1e-6, R+1e-6,1e-6, dim=2
)

gmsh.model.addPhysicalGroup(2, [s[1] for s in bottom_cap], tag=10)
gmsh.model.setPhysicalName(2,10, "bottom") #making it its own physical group w/name

#repeating the process for the top
top_cap = gmsh.model.getEntitiesInBoundingBox(
    -R-1e-6,-R-1e-6,H-1e-6,R+1e-6,R+1e-6,H+1e-6, dim=2
)

gmsh.model.addPhysicalGroup(2,[s[1] for s in top_cap], tag=11)
gmsh.model.setPhysicalName(2,11, "top")

#cylinder walls- we'll get these by grabbing all the entities that are not the caps
all_surfaces = [s[1] for s in gmsh.model.getEntities(2)]
walls = [sid for sid in all_surfaces if sid not in [s[1] for s in bottom_cap]+[s[1] for s in top_cap]]

gmsh.model.addPhysicalGroup(2, walls, tag=12)
gmsh.model.setPhysicalName(2,12,"walls")

#now we generate the mesh
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", dv)
gmsh.model.mesh.generate(3) #3D mesh
gmsh.model.mesh.setOrder(1) # we use 1 for 'linear elements'. mesh order is an important concept.

#now we convert to a DOLFINx mesh

mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank=0, gdim=3)
mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

gmsh.finalize()

V = fem.functionspace(mesh, ("N1curl",1)) #we need vector-valued basis functions; we'll use Nedelec
#perfectly electrically conductive walls; not fully understood, works by trial and error
zero = fem.Function(V)
zero.x.array[:] = 0.0
zero.x.scatter_forward()

pec_facets = facet_tags.indices[(facet_tags.values == 12) | (facet_tags.values == 10) | (facet_tags.values == 11)]
dofs_pec = fem.locate_dofs_topological(V, mesh.topology.dim - 1, pec_facets)

bc_pec = fem.dirichletbc(zero,dofs_pec)

bcs= [bc_pec]
#setting material constants

mu_r = fem.Constant(mesh, float(1.0))
eps_r = fem.Constant(mesh, 1.0)


#introducing physical parameters

mu0 = 4* np.pi*1.e-7 #mag perm
eps0 = 8.85e-12 #free space permit
c0 = 1/ np.sqrt(mu0*eps0) #speed of light baby
freq = 1e5 #initializing frequency at 0.1 MHz; we'll iterate through and solve for each frequency
omega = freq * 2 * np.pi
freq_max = 10e10 #loop terminates at 10 GHz

#defining a small current source which will generate the EM waves in the cavity
def J_expr(x):
    sigma = 0.1 * R
    center_z = 0.0
    r2 = x[0]**2 + x[1]**2 + (x[2]-center_z)**2
    amp = np.exp(-r2/(2*sigma**2))
    return np.stack([amp,np.zeros_like(x[0]),np.zeros_like(x[0])], axis=0)



J_func = fem.Function(V)
J_func.interpolate(J_expr) #generates current source as a function of space

E = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

k0_sq = fem.Constant(mesh, (omega / c0)**2)
omega_c = fem.Constant(mesh, 1j * omega)

print(mu_r.value)
#bilinear
a = (
    ufl.inner(ufl.curl(E), ufl.curl(v)) / mu_r - k0_sq * eps_r *ufl.inner(E, v)
) * ufl.dx
#linear
L = -1j * omega_c *ufl.inner(J_func, v)*ufl.dx


#here's where we solve the equation
problem = LinearProblem( #solving a linear system of equations
    a,
    L,
    bcs=bcs,
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="Resonator",
)
Eh = problem.solve()



topology, cell_types, geometry = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, style="wireframe")
plotter.view_xz()
plotter.show()
print("complete")