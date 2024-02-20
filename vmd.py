import torch
import numpy as np
import matplotlib.pyplot as plt

class VMD2D(torch.nn.Module):
    def __init__(self, alpha, K, signal_size):
        """
        初始化VMD参数
        :param alpha: 正则化参数
        :param K: 分解的模态数量
        :param signal_size: 信号的尺寸
        """
        super(VMD2D, self).__init__()
        self.alpha = alpha
        self.K = K
        self.signal_size = signal_size

        # 初始化模态和频率
        self.u = torch.nn.Parameter(torch.randn(K, *signal_size))
        self.omega = torch.nn.Parameter(torch.randn(K))

    def forward(self, signal):
        """
        前向传播，定义优化问题
        :param signal: 输入信号
        :return: 分解后的模态
        """
        # FFT of the signal
        f_signal = torch.fft.fft2(signal)

        # FFT of the modes
        f_modes = torch.fft.fft2(self.u)

        # Compute the VMD energy
        energy = 0
        for k in range(self.K):
            energy += torch.sum(torch.abs(f_signal - f_modes[k] * torch.exp(-1j * self.omega[k]))**2)

        # Add the total variation term
        gradients = torch.gradient(self.u, spacing=(1.0, 1.0), dim=(0, 1))
        total_variation = sum(torch.sum(torch.abs(g)) for g in gradients)
        energy += self.alpha * total_variation

        return energy

# 创建一个简单的合成图像
def create_synthetic_image(size):
    x = np.linspace(-10, 10, size[0])
    y = np.linspace(-10, 10, size[1])
    X, Y = np.meshgrid(x, y)
    Z = np.sin(1 * X) + np.cos(1 * Y) + np.random.normal(0, 0.1, X.shape)
    return Z

# 可视化函数
def plot_images(images, titles, cmap='gray'):
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# 创建合成图像
signal_size = (256, 256)
synthetic_image = create_synthetic_image(signal_size)
signal = torch.tensor(synthetic_image, dtype=torch.float32)

# 应用VMD
vmd = VMD2D(alpha=2000, K=3, signal_size=signal_size)
optimizer = torch.optim.Adam(vmd.parameters(), lr=0.001)

# 优化循环
for i in range(100000):
    optimizer.zero_grad()
    loss = vmd(signal)
    if i % 100 == 0:
        print(f'Loss: {loss}')
    loss.backward()
    optimizer.step()

# 提取模态
modes = vmd.u.detach().numpy()

# 可视化结果
plot_images([synthetic_image] + list(modes), ['Original Image'] + [f'Mode {i+1}' for i in range(vmd.K)])
modes.shape  # 返回模态的形状以确认结果
