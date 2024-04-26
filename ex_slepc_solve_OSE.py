"""A moderate (complex) generalised eigenproblem 'A@x=c B@x' where omega are
the eigenvalues. 

Solves the following linear eigenvalue problem for values c
that satisfy :
  phi''(y) - alpha^2 phi(y) - psi(y) = 0
  psi''(y) - alpha^2 psi(y) - i alpha Re { ( U(y) - c ) psi(y) - U''(y) } = 0
 subject to phi = \phi' = 0 on y = -1 and +1, where
 alpha = 1.02, Re = 5772.2 and U(y) = 1 - y^2.

 The matrix problem is constructed manually in this case, using second-order
 finite differences. A sparse eigenvalue routine is employed, via SLEPc.
 These values approximately correspond to the first neutral temporal mode
 in plane Poiseuille flow, therefore the test to be satisfied is that an eigenvalue
 exists with very small imaginary part.


Run: python ex_slepc_solve_harmonic.py -st_type sinvert -eps_view
"""
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import sp2petsc

# 101 points across the domain
nnodes = 4001
N = 2 * nnodes
L = 1.0
d = 2 * L / (nnodes - 1)
nodes = np.linspace(-L, L, nnodes)
U = 1 - nodes * nodes
Udd = -2 * np.ones(nnodes)

# plt.plot(nodes,U)
# plt.spy(a,markersize=1)
# plt.show()


alpha = 1.02
Re = 5772.2
eye = complex(0, 1)

# sparse matrix storage
a = sp.sparse.lil_matrix((N, N), dtype=complex)
b = sp.sparse.lil_matrix((N, N), dtype=complex)

# For more complex problems lil_matrix allows for
# efficient incremental construction.

# boundary conditions at the left boundary
a[0, 0] = 1.0  # phi( left ) = 0
a[1, 0] = -1.5 / d  # phi'( left ) = 0
a[1, 2] = 2.0 / d
a[1, 4] = -0.5 / d
# fill the interior nodes
for i in range(1, nnodes - 1):
    # the first equation at the i'th nodal point
    row = 2 * i
    a[row, row] = -2.0 / (d * d) - alpha * alpha
    a[row, row - 2] = 1.0 / (d * d)
    a[row, row + 2] = 1.0 / (d * d)
    a[row, row + 1] = -1.0

    row += 1
    # the second equation at the i'th nodal point
    a[row, row] = -2.0 / (d * d) - alpha * alpha - eye * alpha * Re * U[i]
    a[row, row - 2] = 1.0 / (d * d)
    a[row, row + 2] = 1.0 / (d * d)
    a[row, row - 1] = eye * alpha * Re * Udd[i]

    b[row, row] = -eye * alpha * Re

# boundary conditions at right boundary
a[N - 2, N - 2] = 1.5 / d
a[N - 2, N - 4] = -2.0 / d
a[N - 2, N - 6] = 0.5 / d  # psi'( right ) = 0
a[N - 1, N - 2] = 1.0  # psi( right ) = 0

system = sp2petsc.SLEPcGeneralisedEigenSystem(a, b)
system.linear_eigensolve()

# renormalize the eigenvectors based only on peak |phi|=1
system.eigenvectors[0, :] /= np.linalg.norm(system.eigenvectors[0, ::2], ord=np.inf)
system.eigenvectors[2, :] /= np.linalg.norm(system.eigenvectors[2, ::2], ord=np.inf)
# plot a couple of the eigenvectors for phi
fig, ax = plt.subplots(2)
# slice notation start [index:end index:step]
ax[0].plot(nodes, system.eigenvectors[0, ::2].real, linewidth=2, label="mode 0 real")
ax[0].plot(nodes, system.eigenvectors[0, ::2].imag, linewidth=2, label="mode 0 imag")
ax[0].plot(nodes, system.eigenvectors[2, ::2].real, linewidth=2, label="mode 2 real")
ax[0].plot(nodes, system.eigenvectors[2, ::2].imag, linewidth=2, label="mode 2 imag")
ax[0].legend()

ax[1].plot(system.eigenvalues[:].real, system.eigenvalues[:].imag, "bo")


plt.show()
