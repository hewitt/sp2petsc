"""A small generalised eigenproblem 'A@x=omega B@x' where omega are
the eigenvalues. 

We formulate and solve the harmonic problem

f''(x)+omega f(x) = 0

distretised with a second-order central scheme. The boundary
conditions are included in the matrix problem (making it generalised)
via

f(0) = f(1) = 0.

Run: python ex_slepc_solve_harmonic.py -st_type sinvert -eps_view


"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import sp2petsc

# 101 points across the domain
n = 101
L = 1.0
h = L / (n - 1)
nodes = np.linspace(0, L, n)

a = sp.sparse.lil_matrix((n, n), dtype=float)
b = sp.sparse.lil_matrix((n, n), dtype=float)

# For more complex problems lil_matrix allows for
# efficient incremental construction. In this case we could
# obviously construct explicitly using constant diagonals
a[0, 0] = 1  # boundary condition
for i in range(1, n - 1):
    a[i, i - 1] = 1 / h**2
    a[i, i] = -2 / h**2
    a[i, i + 1] = 1 / h**2
    b[i, i] = -1
a[n - 1, n - 1] = 1  # boundary condition

system = sp2petsc.SLEPcGeneralisedEigenSystem(a, b)
system.linear_eigensolve()

# eigenvalues are n**2 pi**2 for n=1,2,3...
# eigenvectors are sin(n*pi*x)
# plot the first three eigenvectors
fig, ax = plt.subplots()
ax.plot(nodes, system.eigenvectors[0, :].real, linewidth=1)
ax.plot(nodes, system.eigenvectors[1, :].real, linewidth=1)
ax.plot(nodes, system.eigenvectors[2, :].real, linewidth=1)

plt.show()
