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
from finite_element import FE
from make_animation import make_animation
from plotting_functions import convergencePlot

np.set_printoptions(precision=4)

#Plotting all nodes with color coding for special nodes
def plot_fixed_dofs(dof_fixed, nelx, nely, load_node):
    for i in range(nelx):
        for j in range(nely):
            node_index = i * (nely + 1) + j
            if node_index in dof_fixed:
                dof_plot = plt.plot(i, j, 'o', color='red')  # Red for fixed nodes
            elif node_index == load_node:
                dof_plot = plt.plot(i, j, 'o', color='green')  # Green for load nodes
            else:
                dof_plot = plt.plot(i, j, 'o', color='lightgrey')  # Default for other nodes
    plt.show() 

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
        U = FE(nelx, nely, x, penal, lk)  # displacement vector U

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

if __name__ == "__main__":  # execute main with specified parameters
    nelx = 60  # number elements in x axis
    nely = 30  # number elements in y axis
    volfrac = 0.5  # fractional volume to remain after optimization
    penal = 3.0  # penalization factor for intermediate density values
    rmin = 1.5  # prevents checkerboarding and mesh dependancies (filter size)

    # for animation output
    nelx, nely, x_hist, c_hist = topOpt(nelx, nely, volfrac, penal, rmin, n_iter=200)
    anim = make_animation(nelx, nely, x_hist)
    HTML(anim.to_html5_video())
    anim.save("topOpt_HalfMBB.mp4", fps=10, extra_args=["-vcodec", "libx264"])
    # for convergence plot
    convergencePlot(c_hist)