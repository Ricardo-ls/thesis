import numpy as np
import matplotlib.pyplot as plt

def reconstruct_from_relative(rel_traj: np.ndarray) -> np.ndarray:
    """
    rel_traj: shape = (T-1, 2)
    返回从原点出发累加得到的相对轨迹，shape = (T, 2)
    """
    start = np.array([[0.0, 0.0]])
    positions = np.vstack([start, np.cumsum(rel_traj, axis=0)])
    return positions

def main():
    rel_samples = np.load("data_eth_20_rel.npy")
    print("相对位移样本 shape:", rel_samples.shape)

    num_to_plot = min(20, len(rel_samples))
    indices = np.random.choice(len(rel_samples), size=num_to_plot, replace=False)

    plt.figure(figsize=(8, 6))

    for i in indices:
        rel_traj = rel_samples[i]               # shape = (19, 2)
        traj = reconstruct_from_relative(rel_traj)  # shape = (20, 2)

        x = traj[:, 0]
        y = traj[:, 1]

        plt.plot(x, y, marker='o', markersize=2)

    plt.title("Relative Trajectories Reconstructed from Displacements")
    plt.xlabel("relative x")
    plt.ylabel("relative y")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    main()