import numpy as np
import matplotlib.pyplot as plt

# 读取刚才保存的样本
samples = np.load("data_eth_20.npy")

print("samples shape:", samples.shape)

# 随机画前 20 条轨迹
num_to_plot = min(20, len(samples))

plt.figure(figsize=(8, 6))

for i in range(num_to_plot):
    traj = samples[i]   # shape = (20, 2)
    x = traj[:, 0]
    y = traj[:, 1]

    plt.plot(x, y, marker='o', markersize=2)

plt.title("Sample Trajectories from ETH")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis("equal")
plt.show()