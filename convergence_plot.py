import matplotlib.pyplot as plt
import matplotlib.animation as animation

def convergencePlot(c_hist):
    # plot demonstrating convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(c_hist) + 1), c_hist, marker="o", linestyle="-", color="b")
    plt.title("Objective Function Convergence")
    plt.xlabel("Iteration Number")
    plt.ylabel("Objective Function Value")
    plt.grid(True)
    plt.show()
    return 