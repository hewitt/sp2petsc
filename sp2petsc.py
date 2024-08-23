"""This is the sp2petsc module.

This module provides some simplistic boiler plate to allow the user to:

1. call PETSc linear solvers for a matrix problem defined in terms of
SciPy matrices. To do this we simply convert the SciPy (sparse) matrix
to a SEQUENTIAL PETSc format.

2. call SLEPc eigenvalue solvers for a discrete problem defined in
terms of SciPy matrices. Again to do this the SciPy (sparse) matrices
are converted to the PETSc sequential format.
"""

__version__ = "0.1"
__author__ = "Rich Hewitt"

# be sure to pass command line arguments to petsc4py
import sys
import petsc4py
import slepc4py
import time

# these 2 lines need to be here despite it being against the usual code style
petsc4py.init(sys.argv)
slepc4py.init(sys.argv)

from petsc4py import PETSc
from slepc4py import SLEPc
import scipy as sp
import numpy as np


def garbage_cleanup():
    PETSc.garbage_view()
    PETSc.garbage_cleanup()


class PETScSparseLinearSystem:
    def __init__(self, A, b):
        """A class that is instantiated using a scipy sparse SQUARE
        matrix 'A' and an array 'b'. These are converted to their
        SEQUENTIAL PETSc counterparts in the constructor and stored in
        this class. This allows for a subsequent linear 'linear_solve'
        to be called using any of the PETSc methods defined in the
        command line arguments. For example,

        python code.py -ksp_type preonly -pc_type lu \
        -pc_factor_mat_solver_type mumps -mat_mumps_icntl_7 2

        to solve using the MUMPS library. Using no command-line
        arguments defaults to a GMRES method.
        """
        # fail if the A matrix is not scipy sparse matrix
        assert sp.sparse.isspmatrix(A)
        # fail if the A matrix is not square
        assert A.shape[0] == A.shape[1]
        # make sure that the matrix is in compressed row format
        A = A.tocsr()
        # set non-public instance variables
        self._N = A.shape[0]
        self._A = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        self._b = PETSc.Vec().createWithArray(b)
        self._A.setFromOptions()
        self._b.setFromOptions()
        self._ksp = PETSc.KSP().create()
        # set the matrix to be used in the linear system
        self._ksp.setOperators(self._A)  # , self._A)
        # set command line options
        self._ksp.setFromOptions()

    def help(self):
        """Print some information about the instance constructed,
        including the matrix 'A' the right-hand side 'b' and the
        solution method chosen by command-line arguments.
        """
        print("-")
        print("A PETSc linear system with a RHS of")
        print(self._b.getArray())
        print("and a sparse matrix of")
        self._A.view()
        print("Solving this system with: ", self._ksp.getType())
        print("-")

    def linear_solve(self):
        """Performs a linear solve of Ax=b, as configured in the
        constructor and returns the solution 'x' as an array. The
        solution method is as defined in the command-line arguments.
        """
        # create a temp PETSc SEQUENTIAL vector for the solution
        petsc_x = PETSc.Vec().createSeq(self._N)
        # solve the linear system, for the defined RHS
        # solution is in petsc_x
        self._ksp.solve(self._b, petsc_x)
        # return the solution as an array
        return petsc_x.getArray()

    def __del__(self):
        """Destructor is explicitly defined to clean up allocations
        made by PETSc4py.
        """
        PETSc.garbage_cleanup()


class SLEPcGeneralisedEigenSystem:
    """A class for a generalised non-symmetric eigenvalue problem 'A @
    x = omega B @ x' for eigenvalues 'omega' and eigenvectors 'x'. It
    is instantiated from two SciPy sparse matrices A and B. These are
    then converted to their PETSc SEQUENTIAL counterparts in the
    constructor. In many applications 'B' has zero rows so requires
    some problem specification via the command line arguments. For
    example,

    python code.py -ksp_type preonly -pc_type lu \
         -pc_factor_mat_solver_type mumps -mat_mumps_icntl_7 2 \
         -st_type sinvert -eps_nev 3 -eps_target 10 \
         -eps_view

    to make use of the MUMPS linear solver, apply a shift and invert
    transformation, looking for 3 eigenvalues around the target
    location of 'omega=10'. Note: eps_view reports verbose settings.
    """

    def __init__(self, A, B, target=0):
        try:
            # fail if the A or B matrix is not a scipy sparse matrix
            assert sp.sparse.isspmatrix(A)
            assert sp.sparse.isspmatrix(B)
            # fail if the A or B matrix is not square
            assert A.shape[0] == A.shape[1]
            assert B.shape[0] == B.shape[1]
            # fail if the A matrix is different in size to the B matrix
            assert A.shape[0] == B.shape[0]
        except AssertionError:
            print("The SciPi matrices should be square and of the same size.")
        # make sure that the matrix is in compressed row format
        A = A.tocsr()
        B = B.tocsr()
        # set non-public instance variables for the matrices
        self._N = A.shape[0]  # number of rows/cols
        self._A = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
        self._B = PETSc.Mat().createAIJ(size=B.shape, csr=(B.indptr, B.indices, B.data))
        # eigenvalue problem solver
        self._eps = SLEPc.EPS().create()
        self._eps.setOperators(self._A, self._B)
        self._eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        self._eps.setTarget(target)
        self._eps.setFromOptions()  # allow overide from CLI options
        # spectral transformation
        self._st = self._eps.getST()
        self._st.setType(SLEPc.ST.Type.SINVERT)
        # Krylov object
        self._ksp = self._st.getKSP()
        self._ksp.setOperators(self._A, self._B)
        self._ksp.setType(PETSc.KSP.Type.PREONLY)
        # preconditioner
        self._pc = self._ksp.getPC()
        self._pc.setType(PETSc.PC.Type.LU)
        self._pc.setFactorSolverType(PETSc.Mat.SolverType.SUPERLU_DIST)
        self._pc.setFactorSetUpSolverType()

    def linear_eigensolve(self, info=True):
        """Solve the generalised linear eigenvalue problem. The
        resulting array of eigenvalues and eigenvectors are stored as
        instance variables 'omega'. To access these see
        """
        # solve the problem
        self._eps.solve()
        # get the converged state
        self._eps.getConverged()
        # number of converged eigenvalues
        nconv = self._eps.getConverged()
        if info:
            # output some results
            print("[RESULT]")
            print(f"Number of iterations of the method: {self._eps.getIterationNumber()}")
            print(f"Solution method: {self._eps.getType()}")
            # getDimensions returns a tuple, the first entry is the number of requested eigenvalues
            print(f"Number of requested eigenvalues: {self._eps.getDimensions()[0]}")
            tol, maxit = self._eps.getTolerances()
            print(f"Stopping condition: tol={tol}, maxit={maxit}")
            print(f"Number of converged eigenpairs: {nconv}")
        # PETSc storage for the Re and Im part of the eigenvector
        petsc_xr = PETSc.Vec().createSeq(self._N)
        petsc_xi = PETSc.Vec().createSeq(self._N)
        # simple np storage for the eigenvalues
        self.eigenvalues = np.empty(nconv, dtype=complex)
        self.eigenvectors = np.empty((nconv, self._N), dtype=complex)
        EYE = complex(0, 1)
        if nconv > 0:
            print("")
            print("        k          ||Ax-kx||/||kx|| ")
            print("----------------- ------------------")
            for i in range(nconv):
                # getEigenpair: returns the complex eigenvalue, but allocates the eigenvector via xr & xi arguments
                self.eigenvalues[i] = self._eps.getEigenpair(i, petsc_xr, petsc_xi)
                print("eigenvalue:")
                print(self.eigenvalues[i])
                # get the i-th eigenvector from SLEPc and move into an array
                self.eigenvectors[i, :] = petsc_xr.getArray() + petsc_xi.getArray() * EYE
                # normalise the eigenvector so the peak value is 1
                self.eigenvectors[i, :] /= np.linalg.norm(self.eigenvectors[i], ord=np.inf)
                error = self._eps.computeError(i)
                print(" %9f%+9f j  %12g" % (self.eigenvalues[i].real, self.eigenvalues[i].imag, error))
            print("")
