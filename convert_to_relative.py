import numpy as np

def absolute_to_relative(samples: np.ndarray) -> np.ndarray:
    """
    输入:
        samples: shape = (N, T, 2)
    输出:
        rel_samples: shape = (N, T-1, 2)
    """
    rel_samples = samples[:, 1:, :] - samples[:, :-1, :]
    return rel_samples


def main():
    input_path = "data_eth_20.npy"
    output_path = "data_eth_20_rel.npy"

    # 读取绝对坐标轨迹
    samples = np.load(input_path)
    print("原始绝对坐标样本 shape:", samples.shape)

    # 转成相对位移
    rel_samples = absolute_to_relative(samples)
    print("相对位移样本 shape:", rel_samples.shape)

    # 保存
    np.save(output_path, rel_samples)
    print(f"已保存到: {output_path}")

    # 打印一条样本对比
    print("\n第一条绝对坐标样本：")
    print(samples[0])

    print("\n第一条相对位移样本：")
    print(rel_samples[0])


if __name__ == "__main__":
    main()