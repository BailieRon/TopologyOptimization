import numpy as np

# Finite Element Code
def FE(nelx, nely, x, penal, lk):
    KE = lk()  # Global stiffness matrix
    K = np.zeros(((nelx + 1) * (nely + 1) * 2, (nelx + 1) * (nely + 1) * 2))
    F = np.zeros(((nelx + 1) * (nely + 1) * 2, 1))
    U = np.zeros(((nelx + 1) * (nely + 1) * 2, 1))

    # assemble global stiffness matrix
    for elx in range(1, nelx + 1): 
        for ely in range(1, nely + 1):
            n1 = (nely + 1) * (elx - 1) + ely  # upper right element node number for Ue
            n2 = (nely + 1) * elx + ely  # extract element disp from global disp
            edof = np.array(
                [
                    2 * n1 - 1,
                    2 * n1,
                    2 * n2 - 1,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n2 + 2,
                    2 * n1 + 1,
                    2 * n1 + 2,
                ]
            )
            K[np.ix_(edof - 1, edof - 1)] += x[ely - 1, elx - 1] ** penal * KE
    
    F[1, 0] = -1

    # loads and supports
    # identify geometrically constrained nodes from element x and y arrays
    dof_fixed = np.union1d(
        np.arange(0, 2 * (nely + 1), 2), np.array([2 * (nelx + 1) * (nely + 1) - 1])
    )
    # array of nodes from element x and y arrays
    dofs = np.arange(0, 2 * (nelx + 1) * (nely + 1))
    # filter mask to grab free nodes from node list
    dof_free = np.setdiff1d(dofs, dof_fixed)

    # # SOLVER
    U[dof_free] = np.linalg.solve(
        K[np.ix_(dof_free, dof_free)], F[dof_free]
    )  # solve for displacement at free nodes
    U[dof_fixed] = 0  # fix geometrically constrained nodes
    return U