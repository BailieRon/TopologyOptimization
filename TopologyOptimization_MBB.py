import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import animation
from matplotlib import colors as mcolors
from IPython.display import HTML

# import function modules for optimization
from mesh_filter import check
from optimality_criteria import OC
from element_stiffness_2D import lk

np.set_printoptions(precision=4)


# Finite Element Code
def FE(nelx, nely, x, penal):
    KE = lk()  # Global stiffness matrix
    K = np.zeros(((nelx + 1) * (nely + 1) * 2, (nelx + 1) * (nely + 1) * 2))
    F = np.zeros(((nelx + 1) * (nely + 1) * 2, 1))
    U = np.zeros(((nelx + 1) * (nely + 1) * 2, 1))

    # assembly
    for elx in range(1, nelx + 1):  # assemble global stiffness from elemental stiffness
        for ely in range(1, nely + 1):
            n1 = (nely + 1) * (elx - 1) + ely  # upper right element node number for  Ue
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

    # Plotting all nodes (as a structured grid)
    for i in range(nelx + 1):
        for j in range(nely + 1):
            plt.plot(i, j, "o", color="lightgrey")

    # # SOLVER
    U[dof_free] = np.linalg.solve(
        K[np.ix_(dof_free, dof_free)], F[dof_free]
    )  # solve for displacement at free nodes
    U[dof_fixed] = 0  # fix geometrically constrained nodes
    return U


# plotting to visualize fixed and free DOF's
# def plot_fixed_dofs(dof_fixed, nelx, nely):
#     # Your plotting code here, using nelx and nely
#     nodes_x = (dof_fixed // 2) % (nelx + 1)  # X coordinate of the node
#     nodes_y = (dof_fixed // 2) // (nelx + 1)  # Y coordinate of the node
#     # Highlighting fixed nodes
#     plt.plot(nodes_x, nodes_y, 'o', color='red', label='Fixed DoFs')
#     plt.legend()
#     plt.xlabel('X coordinate')
#     plt.ylabel('Y coordinate')
#     plt.title('Fixed Degrees of Freedom in the Mesh')
#     plt.gca().invert_yaxis()  # Invert y-axis to match the typical FEA node layout
#     plt.grid(True)
#     plt.show()
# plot_fixed_dofs(dof_fixed, nelx, nely)


def make_animation(nelx, nely, x_hist):
    x_hist = x_hist[::2]
    fig, ax = plt.subplots()
    im = ax.imshow(-x_hist[0], cmap="gray", animated=True)

    def update_frame(frame):
        x = -x_hist[frame]
        im.set_array(x)
        return (im,)

    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(x_hist),
        blit=True,
    )
    plt.close(fig)
    return anim


def topOpt(nelx, nely, volfrac, penal, rmin, n_iter: int):
    # initialization
    x_hist = []  # Store x for animation
    c_hist = []
    x = np.ones((nely, nelx)) * volfrac  # initialize matrix populated by volfrac,
    # initial material distribution to set element density
    loop = 0  # intialize iterations for optimization
    change = 1.0  # updates iter

    while (
        change > 0.01
    ):  # continues as change > 0.01, at which point convergence is observed
        loop += 1  # iteration counter
        xold = np.copy(x)  # store current x

        if loop > n_iter:
            break

        # FE Analysis
        U = FE(nelx, nely, x, penal)  # displacement vector U

        # Objective function and sensitivity analysis
        KE = lk()
        c = 0.0  # initialize objective function value (compliance) as zero float type
        dc = np.zeros((nely, nelx))  # initialize sensitivity of objection function to 0
        for ely in range(1, nely + 1):  # nested for loop over element y component
            for elx in range(1, nelx + 1):  # nested foor loop over element x component
                # upper left element node number for element displacement Ue
                n1 = (nely + 1) * (elx - 1) + ely
                # upper right element node number for element displacement Ue
                n2 = (nely + 1) * (elx) + ely
                Ue_indices = [
                    2 * n1 - 2,
                    2 * n1 - 1,
                    2 * n2 - 2,
                    2 * n2 - 1,
                    2 * n2,
                    2 * n2 + 1,
                    2 * n1,
                    2 * n1 + 1,
                ]
                Ue = U[Ue_indices]  # Extract the displacement vector for the element
                f_int = np.dot(Ue.T, np.dot(KE, Ue))
                c += (
                    x[ely - 1, elx - 1] ** penal * f_int
                )  # add elemental contribution to objective function
                dc[ely - 1, elx - 1] = (
                    -penal * x[ely - 1, elx - 1] ** (penal - 1) * f_int
                )  # sensitivity calculation of objective function

        c_hist.append(c.item())
        dc = check(nelx, nely, rmin, x, dc)  # filter sensitivies with check function
        x = OC(
            nelx, nely, x, volfrac, dc
        )  # update design variable x based on OC function
        change = np.max(np.abs(x - xold))  # calclulate max value to check convergence
        print(
            f"Iteration: {loop}, Objective: {c.item():.4f}, Volume: {np.mean(x):.4f}, Change: {change:.4f}"
        )

        x_hist.append(x.copy())
    return (nelx, nely, x_hist, c_hist)


def convergencePlot(c_hist):
    # plot demonstrating convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(c_hist) + 1), c_hist, marker="o", linestyle="-", color="b")
    plt.title("Objective Function Convergence")
    plt.xlabel("Iteration Number")
    plt.ylabel("Objective Function Value")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":  # execute main with specified parameters
    nelx = 60  # number elements in x axis
    nely = 30  # number elements in y axis
    volfrac = 0.5  # fractional volume to remain after optimization
    penal = 3.0  # penalization factor for intermediate density values
    rmin = 1.5  # prevents checkerboarding and mesh dependancies (filter size)

    # for animation output
    nelx, nely, x_hist, c_hist = topOpt(nelx, nely, volfrac, penal, rmin, n_iter=200)
    convergencePlot(c_hist)
    anim = make_animation(nelx, nely, x_hist)
    HTML(anim.to_html5_video())
    anim.save("topOpt_HalfMBB.mp4", fps=10, extra_args=["-vcodec", "libx264"])
