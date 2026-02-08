import gmsh
from dolfinx.io import gmsh as gmshio
from dolfinx.fem.petsc import LinearProblem #we are not using an explicit stepping scheme
from mpi4py import MPI
from dolfinx import fem, plot, mesh, fem, la #importing the function space
import numpy as np
from dolfinx.plot import vtk_mesh
import pyvista
import ufl #ufl = 'unified form language'
from dolfinx import default_scalar_type
from petsc4py import PETSc
import dolfinx.fem.petsc as petsc
from pyvista import plotter

import matplotlib as mpl
gmsh.initialize() #turning on gmsh

X = 50.0      # length in x
T = 100.0       #length in time
nx= 50         #Discretization of the interval

domain = mesh.create_interval(MPI.COMM_WORLD, nx, [0.0, X]) #this is how Dolfinx + MPI work to create a mesh

from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    dx,
    grad,
    inner,
    system,
)
V = fem.functionspace(domain, ("CG", 1)) #CG stands for continuous galerkin, '1' possibly linearity. Previous examples used Lagrange.


def boundary(x): #defining the boundary, as in the deflection example. 
    return np.isclose(x[0], 0.0) | np.isclose(x[0], X)

boundary_dofs = fem.locate_dofs_geometrical(V, boundary)

bc_value = fem.Constant(domain, 0.0)
bc = fem.dirichletbc(bc_value, boundary_dofs, V)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
un = fem.Function(V)
#un.interpolate(lambda x: 0.2 * np.exp(-(x[0]-X/2)**2 / (2*5**2))) #setting an initial waveform to subsequently evolve homogenously
unm1 = fem.Function(V)
unm1.interpolate(lambda x: 0.2 * np.exp(-(x[0]-X/2)**2 / (2*5**2)))
c = fem.Constant(domain, 1.0)
dt = fem.Constant(domain, 0.075)
f = fem.Function(V)  # forcing function
#f.interpolate(lambda x: np.exp(-0.5*((x[0]-X/2)/(0.75**2))))

t = 0.0


# Bilinear form a(u, v)
a = (
    u * v * ufl.dx
    + c**2 * dt**2 * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
)

# Linear form L(v)
L = (
    (2 * un - unm1 + dt**2 * f) * v * ufl.dx
)

plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif")

# Initialize uh
uh = fem.Function(V)

# Get the x-coordinates of the mesh
x = domain.geometry.x  # shape (nx+1, 1)

# Create 2D coordinates for PyVista: x along X, solution u along Y, Z=0
coords = np.zeros((x.shape[0], 3))
coords[:, 0] = x[:, 0]       # x-axis
coords[:, 1] = uh.x.array    # initial solution along y
coords[:, 2] = 0             # flat z-axis

# Create a PyVista PolyData object for a 1D curve
grid = pyvista.PolyData(coords)

#grid.point_data["uh"] = uh.x.array

viridis = mpl.colormaps["viridis"]
sargs = dict(
    title_font_size=25,
    label_font_size=20,
    fmt="%.2e",
    color="black",
    position_x=0.1,
    position_y=0.8,
    width=0.8,
    height=0.1,
)

##print("works to here")

renderer = plotter.add_mesh(
    grid,
    show_edges=True,
    lighting=False,
    cmap=viridis,
    scalar_bar_args=sargs,
)


while (t<T):
    problem = LinearProblem( #solving a linear system of equations
    a,
    L,
    bcs=[bc],
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    petsc_options_prefix="OneDwave_",)
    uh = problem.solve()
    # Update plotter
    grid.points[:, 1] = uh.x.array  # update y-coordinate with current solution
    plotter.write_frame()
    unm1.x.array[:] = un.x.array
    un.x.array[:] = uh.x.array
    t += float(dt)


plotter.close()

print("Complete")