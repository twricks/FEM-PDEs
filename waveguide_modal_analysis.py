#following Dolfinx tutorial for the electromagnetic modal analysis of a half-filled waveguide. 

from pathlib import Path #python's module providing an object-oriented way to work with file system paths

from mpi4py import MPI #wrapper for Message Passing Interface, which is written in C/Fortran.

from petsc4py import PETSc #wrapper for Portal Extensible Toolkit for Scientific Computation

import numpy as np
from slepc4py import SLEPc

import ufl 
from basix.ufl import element, mixed_element
from dolfinx import fem, plot
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.mesh import CellType, create_rectangle, exterior_facet_indices, locate_entities

try: 
    import pyvista

    have_pyvista = True

except ModuleNotFoundError:
    print("pyvista and pyvistaqt required")
    have_pyvista = False

if PETSc.IntType == np.int64 and MPI.COMM_WORLD.size > 1:
    print("solver fails with PETSc and 64 bit integers")
    exit(0)

try:
    from dolfinx.io import VTXWriter
    has_vtx = True

except ImportError:
    print("VTXWriter not available, solution will not be saved")
    has_vtx = False
#all imports work

#we define a set of equations for the TM, TE modes

def TMx_condition(
        kx_d: complex, kx_v: complex, eps_d: complex, eps_v:complex, d: float, h: float
) -> float:
    return kx_d / eps_d * np.tan(kx_d *d) + kx_v / eps_v * np.tan(kx_v * (h-d))

def TEx_condition(
        kx_d: complex, kx_v: complex, d: float, h:float) -> float:
    return kx_d / np.tan(kx_d * d) +kx_v / np.tan(kx_v * (h-d))

#next, the verify_mode function checks whether a certain k_z satisfy the equations within some threshold (close to zero in the last equations)

def verify_mode(
        kz: complex,
        w: float,
        h: float, 
        d: float, 
        lmbd0: float,
        eps_d: complex,
        eps_v: complex, 
        threshold: float, 
) -> np.bool_:
    k0 = 2 * np.pi / lmbd0
    ky = np.pi / w #assume n=1
    kx_d_target = np.sqrt(k0**2 * eps_d - ky**2 + -(kz**2)+0j)
    alpha = kx_d_target**2
    beta = alpha -k0**2 * (eps_d - eps_v)
    kx_v = np.sqrt(beta)
    kx_d = np.sqrt(alpha)
    f_tm = TMx_condition(kx_d, kx_v, eps_d, eps_v, d, h)
    f_te = TEx_condition(kx_d, kx_v, d, h)
    return np.isclose(f_tm, 0, atol=threshold) or np.isclose(f_te, 0, atol = threshold)

#creating a simple rectangular domain

w = 1
h= 0.45*w 
d = 0.5*h 

nx=100
ny = int(0.4* nx)

msh = create_rectangle(MPI.COMM_WORLD, np.array([[0, 0], [w,h]]),(nx,ny), CellType.quadrilateral)
msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)

eps_v = 1
eps_d = 2.45

D = fem.functionspace(msh, ("DQ",0))
eps = fem.Function(D)

cells_d = locate_entities(msh,msh.topology.dim, lambda x: x[1] <= d)

eps.x.array[:] = eps_v
eps.x.array[cells_d] = eps_d

#specifying the elements
degree = 1
RTCE = element("RTCE", msh.basix_cell(), degree, dtype=PETSc.RealType)
Q = element("Lagrange", msh.basix_cell(), degree, dtype = PETSc.RealType)
V = fem.functionspace(msh, mixed_element([RTCE, Q])
)

#defining the weak form, with lambda / 0.2
lmbd0 = h / 0.2
k0 = 2*np.pi / lmbd0

et, ez = ufl.TrialFunctions(V)
vt, vz = ufl.TestFunctions(V)

a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt))-(k0**2)*eps*ufl.inner(et, vt))*ufl.dx
b_tt = ufl.inner(et, vt)*ufl.dx
b_tz = ufl.inner(et, ufl.grad(vz))*ufl.dx
b_zt = ufl.inner(ufl.grad(ez),vt)*ufl.dx
b_zz = (ufl.inner(ufl.grad(ez),ufl.grad(vz))-(k0**2)*eps*ufl.inner(ez,vz))*ufl.dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz +b_zt +b_zz)

#adding PEC walls
bc_facets = exterior_facet_indices(msh.topology)
bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim-1, bc_facets)
u_bc = fem.Function(V)

with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)

#now we solve the problem using SLEPc

A = assemble_matrix(a, bcs=[bc])
A.assemble()
B = assemble_matrix(b, bcs=[bc])
B.assemble()

#initialize the solver
eps = SLEPc.EPS().create(msh.comm)

#pass this to the matrices using setOperators

eps.setOperators(A,B)

#if the matrices in the problem had known properties, e.g. hermiticity, we could specify this further with setProblemType
#however there are no known special properties here, so we just specify it as a Generalized Non Hermitian eigenvalue problem

eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

tol = 1e-6 #tolerance
eps.setTolerances(tol=tol, max_it=1000)

#specifying Krylov-Schur

eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

#spectral transformation accelerates the calculation of solutions

st = eps.getST()

st.setType(SLEPc.ST.Type.SINVERT)

#setting a target for the spectral transformation

eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

#setting a target value close to k0
eps.setTarget(-((0.5*k0)**2))

#specification of how many eigenvalues we want
eps.setDimensions(nev=1)

eps.solve()

print("Converged eigenpairs:", eps.getConverged())
for i in range(eps.getConverged()):
    print("Residual:", eps.computeError(i))
eps.errorView()

#save the kz

vals = [(i, np.sqrt(-eps.getEigenvalue(i))) for i in range (eps.getConverged())]

#sort kz by real part
vals.sort(key=lambda x: x[1].real)

eh = fem.Function(V)

kz_list = []

out_folder = Path("out_half_loaded_waveguide")
out_folder.mkdir(parents=True, exist_ok=True)

for i, kz in vals:
    #save eigenvector in eh
    eps.getEigenpair(i, eh.x.petsc_vec)

    #computer error for ith eigenvalue
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)

    #verify, save, visualize
    if error < 1e-6 and np.isclose(kz.imag, 0, atol = tol):
        kz_list.append(kz)

        #verify if kz is consistent with the analytical equations
        assert verify_mode(kz, w, h, d, lmbd0, eps_d, eps_v, threshold=1e-2)

        print(f"eigenvalues: {-(kz**2)}")
        print(f"kz/k0: {kz / k0}")

        eh.x.scatter_forward()

        eth, ezh = eh.split()
        ez = eh.sub(1).collapse()

        #transform
        eth.x.array[:] = eth.x.array[:] / kz
        ezh.x.array[:] = eth.x.array

        gdim = msh.geometry.dim

        V_dg = fem.functionspace(msh,("DQ", degree, (gdim,)))
        Et_dg = fem.Function(V_dg)
        Et_dg.interpolate(eth)

        if has_vtx:
            #saving solutions
            with VTXWriter(msh.comm, out_folder / f"sols/Et_{i}.bp", Et_dg) as f:
                f.write(0.0)

            with VTXWriter(msh.comm, out_folder /f"sols/Ez_{i}.bp",ezh) as f:
                f.write(0.0)

        #visualization with Pyvista

        if have_pyvista:
            V_cells, V_types, V_x = plot.vtk_mesh(V_dg)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            Et_values = np.zeros((V_x.shape[0],3), dtype=np.float64)
            Et_values[:,: msh.topology.dim] = Et_dg.x.array.reshape(
                V_x.shape[0], msh.topology.dim
            ).real

            V_grid.point_data["u"]=Et_values

            plotter = pyvista.Plotter()
            plotter.add_mesh(V_grid.copy(),show_edges=False)
            plotter.view_xy()
            plotter.link_views()
            if pyvista.OFF_SCREEN:
                plotter.screenshot(out_folder / "Et.png", window_size=[400,400])
            else:
                plotter.show()

        if have_pyvista:
            V_lagr, lagr_dofs = V.sub(1).collapse()
            V_cells, V_types, V_x = plot.vtk_mesh(V_lagr)
            V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
            V_grid.point_data["u"] = ezh.x.array.real[lagr_dofs]
            plotter = pyvista.Plotter()
            plotter.add_mesh(V_grid.copy(), show_edges=False)
            plotter.view_xy()
            plotter.link_views()
            if pyvista.OFF_SCREEN:
                plotter.screenshot(out_folder / "Ez.png", window_size=[400,400])

            else: 
                plotter.show()

