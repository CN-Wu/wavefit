from wavefit.core.phase_retrieval import multi_plane_gs
import cupy as cp
import matplotlib.pyplot as plt

# ========== 参数 ==========
N = 512  # 采样点
dx = 5e-6  # 采样间距
wavelength = 532e-9  # 波长
z_list = [0.0, 0.01, 0.02]  # 三个传播平面（单位：m）

# ========== 坐标网格 ==========
x = cp.linspace(-N / 2, N / 2 - 1, N) * dx
X, Y = cp.meshgrid(x, x)
r = cp.sqrt(X**2 + Y**2)

# ========== 定义近场高斯光束 + 指定相位 ==========
w0 = 0.4e-3  # 高斯半径
amp = cp.exp(-((r / w0) ** 2))

# 生成相位（例如球差） φ = α * r^4
alpha = 1e9  # 球差强度，可调整
phase = alpha * r**4
field0 = amp * cp.exp(1j * phase)


# ========== 定义角谱传播函数 ==========
def angular_spectrum_propagate(E0, z, wavelength, dx):
    N = E0.shape[0]
    k = 2 * cp.pi / wavelength
    fx = cp.fft.fftfreq(N, d=dx)
    FX, FY = cp.meshgrid(fx, fx)
    H = cp.exp(
        1j * k * z * cp.sqrt(1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2)
    )
    E1 = cp.fft.ifft2(cp.fft.fft2(E0) * cp.fft.fftshift(H))
    return E1


# ========== 传播并生成 I_meas_list ==========
I_meas_list = []
for z in z_list:
    E_z = angular_spectrum_propagate(field0, z, wavelength, dx)
    I_meas_list.append(cp.abs(E_z) ** 2)

# ========== 可视化 ==========
fig, axs = plt.subplots(1, len(z_list), figsize=(15, 4))
for i, z in enumerate(z_list):
    axs[i].imshow(
        cp.asnumpy(cp.log10(I_meas_list[i] + 1e-8)),
        cmap="inferno",
        extent=[x[0] * 1e3, x[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3],
    )
    axs[i].set_title(f"z = {z*1e3:.1f} mm")
    axs[i].set_xlabel("x [mm]")
    axs[i].set_ylabel("y [mm]")
plt.tight_layout()
plt.show()
