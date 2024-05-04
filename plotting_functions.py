import matplotlib.pyplot as plt

def convergencePlot(c_hist):
    # plot demonstrating convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(c_hist) + 1), c_hist, marker="o", linestyle="-", color="b")
    plt.title("Objective Function Convergence")
    plt.xlabel("Iteration Number")
    plt.ylabel("Objective Function Value")
    plt.grid(True)
    plt.show()

def plot_nodes(nelx, nely, dof_fixed):
    fig, ax = plt.subplots()
    for i in range(nelx + 1):
        for j in range(nely + 1):
            # Calculate node index for the flat array structure
            node_index_x = 2 * (i * (nelx + 1) + j)     # DOF index for x-direction
            node_index_y = node_index_x + 1             # DOF index for y-direction

            if node_index_x in dof_fixed or node_index_y in dof_fixed:
                color = 'red'  # Fixed node
            else:
                color = 'gray'  # Free node

            ax.plot(i, j, 'o', color=color)

    ax.set_aspect('equal', adjustable='box')
    plt.title('Finite Element Mesh Nodes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()