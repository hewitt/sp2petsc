"""A very small matrix solve of 'a@x=b' using the boiler-plate code
provided by sp2petsc.

To switch to (for example MUMPS) as a linear solver:

python petsc_example.py  -ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -mat_mumps_icntl_7 2

"""
import numpy as np
import scipy as sp
import sp2petsc

# make a SciPy sparse matrix for the 'a' matrix
a = sp.sparse.lil_matrix((2, 2), dtype=float)
# an array for the RHS vector 'b'
b = np.ones(2)

a[0, 0] = 1
a[0, 1] = 2
a[1, 0] = 3
a[1, 1] = 4

# convert the SciPy problem to a SEQUENTIAL PETSc problem
system = sp2petsc.PETScSparseLinearSystem(a, b)

# print some helper information after instantiation
system.help()

# solver the PETSc problem
x = system.linear_solve()

# solution should be [-1,1]
print(f"The solution is {x}")
error = sp.linalg.norm(a @ x - b, ord=np.inf)
print(f"The error is {error}")
assert abs(error) < 1.0e-8
