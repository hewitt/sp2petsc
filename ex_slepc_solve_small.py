"""A test 2x2 eigenproblem 'A@x=omega B@x' where omega are
the eigenvalues. Depsite calling a generalised eigenvalue routine,
we'll pass an identity matrix for 'B'.

Run: python ex_slepc_solve_small.py -eps_view
"""
import scipy as sp
import numpy as np
import sp2petsc

a = sp.sparse.lil_matrix((2, 2), dtype=float)
b = sp.sparse.lil_matrix((2, 2), dtype=float)

a[0, 0] = 1
a[0, 1] = 2
a[1, 0] = 4
a[1, 1] = 3
b[0, 0] = 1
b[1, 1] = 1

system = sp2petsc.SLEPcGeneralisedEigenSystem(a, b)
system.linear_eigensolve()
# eigenvalues are -1 and 5.
error = np.linalg.norm(np.sort(system.eigenvalues) - [-1, 5], ord=np.inf)
print(f"{error=}")
assert error < 1.0e-14
