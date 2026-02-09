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
Y = 50.0
T = 100.0       #length in time
t = 0.0  #time initialization
nx= 50         #Discretization of the interval
ny= 50

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [X, Y]], [nx,ny]) #this is how Dolfinx + MPI work to create a mesh

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
V = fem.functionspace(domain, ("CG", 1)) #CG stands for continuous Galerkin, '1' possibly linearity. Previous examples used Lagrange.


def boundary(x): #defining the boundary, as in the deflection example. 
    return np.isclose(x[0], 0.0) | np.isclose(x[0], X) | np.isclose(x[1], 0.0)| np.isclose(x[1], Y)

boundary_dofs = fem.locate_dofs_geometrical(V, boundary)

bc_value = fem.Constant(domain, 0.0)
bc = fem.dirichletbc(bc_value, boundary_dofs, V)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
un = fem.Function(V)
#un.interpolate(lambda x: 0.2 * np.exp(-(x[0]-X/2)**2 / (2*5**2))) #setting an initial waveform to subsequently evolve homogenously
unm1 = fem.Function(V)
unm1.interpolate(lambda x: 5 * np.exp(-((x[0]-1)**2 + (x[1]-1)**2) / (2 * 0.5**2)))
c = fem.Constant(domain, 0.75)
dt = fem.Constant(domain, 0.075)
f = fem.Function(V)  # forcing function
#f.interpolate(lambda x: 5 * np.exp(-((x[0]-1)**2 + (x[1]-1)**2) / (2 * 0.5**2)))

#def f_function(t):
    #return lambda x: x[1] * 0.0023*np.cos(0.25*t)

#f.interpolate(f_function(t))



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

tdim = domain.topology.dim


plotter = pyvista.Plotter()
plotter.open_gif("u_time.gif")

##print("works to here")

topology, cells, geometry = plot.vtk_mesh(V)
uh = fem.Function(V)
grid = pyvista.UnstructuredGrid(topology, cells, geometry)
grid.point_data["uh"] = uh.x.array

coolwarm = mpl.colormaps["coolwarm"]
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
    cmap=coolwarm,
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
    # Update mesh geometry in z-direction
    xyz = geometry.copy()
    xyz[:, 2] = uh.x.array
    grid.points = xyz
    grid.point_data["uh_color"] = uh.x.array
    plotter.update_scalars(grid.point_data["uh_color"], render=True)
    plotter.write_frame()
    unm1.x.array[:] = un.x.array
    un.x.array[:] = uh.x.array
    #f.interpolate(f_function(t))
    t += float(dt)


plotter.close()

print("Complete")