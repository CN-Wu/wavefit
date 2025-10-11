import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import gaussian_filter
from wavefit.core.propagation import angular_spectrum_propagation
from wavefit.core.phase_retrieval import multi_plane_gs
import matplotlib.pyplot as plt

def test_multiplane_gs():
    # ---------------- 参数 ----------------
    N = 256
    dx = 5e-6
    lam = 532e-9
    z_list = [-8e-3, -4e-3, 0.0, 4e-3, 8e-3]  # 多平面传播距离 (相对重建面)
    
    # ---------------- 构造近场 ----------------
    x = (cp.arange(N) - N//2) * dx
    X, Y = cp.meshgrid(x, x)
    R = cp.sqrt(X**2 + Y**2)
    zernike_radius = 0.3e-3
    rho = R / zernike_radius
    theta = cp.arctan2(Y, X)

    # 振幅和相位（高斯 + Zernike astig + sph + 随机平滑）
    amp = cp.exp(-((X**2 + Y**2)/(0.2e-3)**2))
    astig = rho**2 * cp.cos(2*theta)
    sph = rho**4
    rng = cp.random.RandomState(1)
    phi = gaussian_filter(rng.standard_normal((N,N)), sigma=3) * 0
    E0 = amp * cp.exp(1j*(phi + 1*astig + 0.5*sph))

    # ---------------- 生成多平面强度 ----------------
    I_meas_list = []
    for z in z_list:
        Ez, _ = angular_spectrum_propagation(E0, dx, lam, z, pad_factor=2)
        I_meas_list.append(cp.abs(Ez)**2)

    # ---------------- multi-plane GS 恢复 ----------------
    E_rec, residuals = multi_plane_gs(
        I_meas_list,
        z_list,
        dx,
        lam,
        n_iter=5000,
        verbose=True
    )

    # ---------------- 验证 ----------------
    # 传播到各测量面
    Ez_rec_list = [angular_spectrum_propagation(E_rec, dx, lam, z, pad_factor=2)
                   for z in z_list]

    # 近场误差
    rel_err_near = cp.linalg.norm(E_rec - E0) / (cp.linalg.norm(E0)+1e-16)
    print(f"Relative error at reconstruction plane: {float(rel_err_near):.3e}")

    # 可视化
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    axs[0,0].imshow(cp.asnumpy(cp.abs(E0)), cmap='jet'); axs[0,0].set_title("True amplitude (near)")
    axs[0,1].imshow(cp.asnumpy(cp.angle(E0)), cmap='twilight'); axs[0,1].set_title("True phase (near)")
    axs[0,2].imshow(cp.asnumpy(cp.abs(I_meas_list[len(z_list)//2])), cmap='jet'); axs[0,2].set_title("Measured center plane amplitude")

    axs[1,0].imshow(cp.asnumpy(cp.abs(E_rec)), cmap='jet'); axs[1,0].set_title("Recovered amplitude (near)")
    axs[1,1].imshow(cp.asnumpy(cp.angle(E_rec)), cmap='twilight'); axs[1,1].set_title("Recovered phase (near)")
    axs[1,2].imshow(cp.asnumpy(cp.angle(E_rec*cp.conj(E0))), cmap='RdBu'); axs[1,2].set_title("Phase difference")

    plt.tight_layout()
    plt.show()

    # 残差曲线
    plt.figure()
    plt.plot(residuals)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Mean relative error")
    plt.title("Multi-plane GS residuals")
    plt.show()

if __name__ == "__main__":
    test_multiplane_gs()
