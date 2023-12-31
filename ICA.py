import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

# 生成混合信号
np.random.seed(0)
time = np.linspace(0, 5, 1000)
signal = np.sin(2 * time)  # 信号源
noise = np.random.normal(size=signal.shape)  # 噪声
s1 = signal + noise
s2 = 0.5 * signal + noise

# 执行独立成分分析
ica = FastICA(n_components=2)
S = np.c_[s1, s2]
S = ica.fit_transform(S)

# 绘制混合信号和分离信号
plt.figure()

plt.subplot(4, 1, 1)
plt.title('Signal Source s1')
plt.plot(s1)

plt.subplot(4, 1, 2)
plt.title('Signal Source s2')
plt.plot(s2)

plt.subplot(4, 1, 3)
plt.title('Separated Signal')
plt.plot(S[:, 0])

plt.subplot(4, 1, 4)
plt.title('Separated Noise')
plt.plot(S[:, 1])

plt.tight_layout()
plt.show()