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
    

# plt.figure()
# X, Y = np.meshgrid(range(nelx+1), range(nely+1))
# plt.pcolormesh(X, Y, x, cmap='gray', edgecolor='k', shading='flat')
# plt.scatter(X.ravel()[dof_fixed//2 % (nelx+1)], Y.ravel()[dof_fixed//2 // (nelx+1)], color='red', s=50)
# plt.scatter([0], [0], color='green', s=50) # force is applied at the top left
# plt.gca().set_aspect('equal', adjustable='box')
# plt.colorbar()
# plt.show()