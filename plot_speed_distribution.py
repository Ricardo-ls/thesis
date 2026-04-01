import numpy as np
import matplotlib.pyplot as plt

def main():
    rel_samples = np.load("data_eth_20_rel.npy")
    print("相对位移样本 shape:", rel_samples.shape)

    # 每一步速度 = sqrt(dx^2 + dy^2)
    speeds = np.linalg.norm(rel_samples, axis=2)   # shape = (N, T-1)

    all_speeds = speeds.flatten()

    print("速度均值:", all_speeds.mean())
    print("速度标准差:", all_speeds.std())
    print("速度最小值:", all_speeds.min())
    print("速度最大值:", all_speeds.max())

    plt.figure(figsize=(8, 5))
    plt.hist(all_speeds, bins=30)
    plt.title("Speed Distribution")
    plt.xlabel("speed")
    plt.ylabel("count")
    plt.grid(True)
    plt.show()

    

if __name__ == "__main__":
    main()

    import numpy as np

rel_samples = np.load("data_eth_20_rel.npy")
speeds = np.linalg.norm(rel_samples, axis=2).flatten()

print("总步数:", len(speeds))
print("speed < 0.01 的比例:", np.mean(speeds < 0.01))
print("speed < 0.05 的比例:", np.mean(speeds < 0.05))
print("speed < 0.10 的比例:", np.mean(speeds < 0.10))
print("speed < 0.20 的比例:", np.mean(speeds < 0.20))