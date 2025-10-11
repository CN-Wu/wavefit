import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from wavefit.core.propagation import angular_spectrum_propagation
import matplotlib.pyplot as plt
from wavefit.core.phase_retrieval import gerchberg_saxton


def test_gs_with_asa():
    # ---------------- 参数 ----------------
    N = 256 // 4
    dx = 5e-6 * 4
    lam = 532e-9
    z = 6e-2  # 传播距离

    # ---------------- 构造近场 ----------------
    x = (cp.arange(N) - N // 2) * dx
    X, Y = cp.meshgrid(x, x)
    R = cp.sqrt(X**2 + Y**2)
    zernike_radius = 0.2e-3
    rho = R / zernike_radius
    theta = cp.arctan2(Y, X)
    phase_corp_mask = R < zernike_radius

    amp = cp.exp(-((X**2 + Y**2) / (0.1e-3)**2))
    astig = (rho**2) * cp.cos(2 * theta)
    sph = rho**4

    rng = cp.random.RandomState(1)
    phi = gaussian_filter(rng.standard_normal((N, N)), sigma=3)

    phi += 1 * astig + 0.5 * sph
    phi *= phase_corp_mask  # zero outside aperture
    E0 = amp * cp.exp(1j * phi)

    # ---------------- 前向传播生成测量数据 ----------------
    Ez = angular_spectrum_propagation(E0, dx, lam, z, pad_factor=2)
    I_near = cp.abs(E0)**2
    I_far = cp.abs(Ez)**2

    # ---------------- GS 相位恢复 ----------------
    E_recon = gerchberg_saxton(I_near, I_far, dx, lam, z, pad_factor=2, n_iter=500)

    # ---------------- 反向传播验证 ----------------
    Ez_recon = angular_spectrum_propagation(E_recon, dx, lam, z, pad_factor=2)

    # ---------------- 结果对比 ----------------
    rel_err_near = cp.linalg.norm(E_recon - E0) / (cp.linalg.norm(E0) + 1e-16)
    rel_err_far = cp.linalg.norm(Ez_recon - Ez) / (cp.linalg.norm(Ez) + 1e-16)

    print(f"\nRelative error (near field): {float(rel_err_near):.3e}")
    print(f"Relative error (far field): {float(rel_err_far):.3e}")

    # ---------------- 可视化 ----------------
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0, 0].imshow(cp.asnumpy(cp.abs(E0)), cmap='jet')
    axs[0, 0].set_title("True amplitude (near)")
    axs[0, 1].imshow(cp.asnumpy(cp.angle(E0)), cmap='twilight')
    axs[0, 1].set_title("True phase (near)")
    axs[0, 2].imshow(cp.asnumpy(cp.abs(Ez)), cmap='jet')
    axs[0, 2].set_title("Amplitude (far)")

    axs[1, 0].imshow(cp.asnumpy(cp.abs(Ez_recon)), cmap='jet')
    axs[1, 0].set_title("Recovered amplitude (near)")
    axs[1, 1].imshow(cp.asnumpy(cp.angle(E_recon)), cmap='twilight')
    axs[1, 1].set_title("Recovered phase (near)")
    axs[1, 2].imshow(cp.asnumpy(cp.angle(E_recon * cp.conj(E0))), cmap='RdBu')
    axs[1, 2].set_title("Phase difference")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_gs_with_asa()
