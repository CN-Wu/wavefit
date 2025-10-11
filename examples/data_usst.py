import os
import sys
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from wavefit.io.hdf5 import read_hdf5_dataset
from cupyx.scipy.ndimage import gaussian_filter
# from skimage.metrics import structural_similarity as ssim   # pip install scikit-image

# 将 hartmann 包路径加入 sys.path
src_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(src_path, "../../hartmann")))
try:
    from hartmann.centroid.detection import detect_centroids_from_image
except ImportError:
    raise

# 导入你的 multi_plane_gs 和 ASA 传播函数
from wavefit.core.propagation import angular_spectrum_propagation
from wavefit.core.phase_retrieval import multi_plane_gs

plt.style.use("../waveoptics/waveoptics/utils/styles/sci.mplstyle")

# ---------------- 数据加载 ----------------
def dataloader(file_path, dataset_name="BG_DATA/1/DATA"):
    # Camera information: pixel size 3.69 um, 1928x1448
    data = read_hdf5_dataset(file_path, dataset_name).reshape([1448, 1928])
    return data

# ---------------- 数据裁剪 + 质心检测 ----------------
from scipy.ndimage import median_filter

def preprocess_data(
    file_path, 
    crop_x=(160-20, 220+20), 
    crop_y=(1355-20, 1415+20),
    background_method='constant',  # 可选 'median', 'percentile', 'mean'
    background_value=1.4e8
):
    data = dataloader(file_path)
    
    # Step 1: 裁剪
    data_crop = data[crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]
    
    # Step 2: 背景估计与扣除
    if background_method == 'median':
        # 用中值滤波估计平滑背景
        background = median_filter(data_crop, size=background_value)
    elif background_method == 'percentile':
        # 用分位数估计背景
        background = np.percentile(data_crop, 60)
        print(background)
    elif background_method == 'mean':
        background = np.mean(data_crop)
    elif background_method == "constant":
        background = background_value
    else:
        raise ValueError("background_method must be 'median', 'percentile', or 'mean'")
    
    # 扣除背景
    data_bg_removed = data_crop - background
    data_bg_removed[data_bg_removed < 0] = 0  # 避免负值

    # Step 3: 质心检测
    centroid, _, _ = detect_centroids_from_image(data_bg_removed)
    xi, xj = centroid[0]

    return data_bg_removed, xi, xj



def fourier_upsample_gpu(I, up=2, dtype=cp.complex128):
    """
    Fourier-domain zero-padding upsample of a 2D real image I.
    - I: numpy or cupy 2D real array (Ny, Nx)
    - up: upsampling factor (integer)
    Returns: cupy float64 2D array of shape (up*Ny, up*Nx)
    Notes:
      - This implements band-limited sinc interpolation (zero-pad in freq domain).
      - Use up>=1. Keep memory in mind: size grows as up^2.
    """
    I_cp = cp.asarray(I)
    Ny, Nx = I_cp.shape
    Ny2, Nx2 = int(Ny*up), int(Nx*up)

    # FFT of input (complex)
    F = cp.fft.fft2(I_cp.astype(dtype))
    # shift so DC in center for easy cropping/zero-pad (use fftshift convention)
    Fshift = cp.fft.fftshift(F)

    # Prepare zero-padded freq grid centered
    pad_y1 = (Ny2 - Ny)//2
    pad_y2 = Ny2 - Ny - pad_y1
    pad_x1 = (Nx2 - Nx)//2
    pad_x2 = Nx2 - Nx - pad_x1

    Fpad = cp.pad(Fshift, ((pad_y1, pad_y2), (pad_x1, pad_x2)), mode='constant', constant_values=0)
    # inverse shift
    Fpad_ifftshift = cp.fft.ifftshift(Fpad)
    # inverse fft to get upsampled image (complex; take real part)
    I_up = cp.fft.ifft2(Fpad_ifftshift)
    I_up_real = cp.real(I_up)
    # small negative numerical values clamp
    I_up_real = cp.clip(I_up_real, a_min=0.0, a_max=None)
    return I_up_real

def upsample_meas_list(I_meas_list, up=2):
    """
    I_meas_list: list of 2D numpy or cupy arrays (same shape)
    up: integer upsampling factor
    Returns: list of cupy arrays upsampled
    """
    return [fourier_upsample_gpu(I, up=up) for I in I_meas_list]


_eps = 1e-16

def _to_numpy(x):
    """If x is cupy array, convert to numpy; else return as np.array."""
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)

def pearson_corr(I_rec, I_ref, mask=None):
    I_rec = _to_numpy(I_rec).ravel()
    I_ref = _to_numpy(I_ref).ravel()
    if mask is not None:
        m = np.asarray(mask).ravel().astype(bool)
        I_rec = I_rec[m]; I_ref = I_ref[m]
    I_rec = I_rec - I_rec.mean()
    I_ref = I_ref - I_ref.mean()
    denom = (np.linalg.norm(I_rec) * np.linalg.norm(I_ref)) + _eps
    return float(np.dot(I_rec, I_ref) / denom)

# def ssim_index(I_rec, I_ref, mask=None, data_range=None):
#     I_rec = _to_numpy(I_rec); I_ref = _to_numpy(I_ref)
#     if mask is not None:
#         # compute SSIM only on masked bounding box for speed/meaningfulness
#         m = np.asarray(mask).astype(bool)
#         ys, xs = np.where(m)
#         if ys.size == 0:
#             return 0.0
#         y0, y1 = ys.min(), ys.max()+1
#         x0, x1 = xs.min(), xs.max()+1
#         I_r = I_rec[y0:y1, x0:x1]; I_f = I_ref[y0:y1, x0:x1]
#     else:
#         I_r, I_f = I_rec, I_ref
#     if data_range is None:
#         data_range = I_f.max() - I_f.min() + _eps
#     s = ssim(I_f, I_r, data_range=data_range)
#     return float(s)

# ---------------- 可视化 ----------------
def visualize_reconstruction(E_rec):
    amp = cp.abs(E_rec)
    phase = cp.angle(E_rec)

    plt.figure(figsize=(3.25,1.55), constrained_layout=True)
    plt.subplot(1,2,1)
    plt.imshow(cp.asnumpy(amp), cmap='inferno')
    plt.colorbar()
    plt.title("Recovered Amplitude")

    plt.subplot(1,2,2)
    plt.imshow(cp.asnumpy(phase), cmap='jet')
    plt.colorbar()
    plt.title("Recovered Phase")

    plt.show()


def compare_intensity_evolution(reconstructed_field, I_meas_list, z_list, wavelength, dx):
    """
    Compare reconstructed vs measured intensities across all planes,
    and also visualize the reconstructed phase for each plane.

    Parameters
    ----------
    reconstructed_field : cp.ndarray
        The complex field reconstructed at the near plane (GS result).
    I_meas_list : list of cupy.ndarray
        Measured intensity images for each z.
    z_list : list of float
        Propagation distances (same as used in GS).
    wavelength : float
        Wavelength in meters.
    dx : float
        Sampling pitch in meters.
    """

    z_list = np.array(z_list)
    z_list = z_list - z_list.min()  # relative to first plane
    n_planes = len(z_list)

    # 每一行显示一个z平面：3列 -> [重建强度, 测量强度, 重建相位]
    fig, axes = plt.subplots(n_planes, 3, figsize=(4.5, 1.6 * n_planes), constrained_layout=True)

    if n_planes == 1:
        axes = axes[None, :]  # 保证二维索引

    for i, z in enumerate(z_list):
        # --- propagate reconstructed field to this plane ---
        Ez, _ = angular_spectrum_propagation(reconstructed_field, dx, wavelength, z)
        I_prop = cp.abs(Ez) ** 2
        phase = cp.angle(Ez)

        I_prop = I_prop.get()
        phase = phase.get()
        I_meas = I_meas_list[i].get()

        # --- normalize intensity for visualization ---
        I_prop /= I_prop.max()
        I_meas /= I_meas.max()

        # --- compute NMSE for quantitative comparison ---
        nmse = np.mean((I_prop - I_meas) ** 2)

        # --- plot reconstructed intensity ---
        ax_amp = axes[i, 0]
        im0 = ax_amp.imshow(I_prop, cmap="inferno")
        ax_amp.set_title(f"Recon. Intensity\nz={z*1e3:.2f} mm", fontsize=8)
        ax_amp.axis("off")
        plt.colorbar(im0, ax=ax_amp, fraction=0.046, pad=0.04)

        # --- plot measured intensity ---
        ax_meas = axes[i, 1]
        im1 = ax_meas.imshow(I_meas, cmap="inferno")
        ax_meas.set_title(f"Measured\nNMSE={nmse:.2e}", fontsize=8)
        ax_meas.axis("off")
        plt.colorbar(im1, ax=ax_meas, fraction=0.046, pad=0.04)

        # --- plot reconstructed phase ---
        ax_phase = axes[i, 2]
        im2 = ax_phase.imshow(phase, cmap="jet", vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title("Reconstructed Phase", fontsize=8)
        ax_phase.axis("off")
        plt.colorbar(im2, ax=ax_phase, fraction=0.046, pad=0.04)

    plt.suptitle("Reconstruction Comparison across Propagation Planes", fontsize=10)
    plt.savefig("./compare.jpg")
    plt.close()


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    # 文件列表（按 z 顺序排列）
    file_path_list = [f"./data/1030nm_Lens40cm_{i:.1f}.lbp2Data" for i in np.arange(10.5, 15.0, 1)]

    dx = 3.69e-6  # 像素间距
    wavelength = 1030e-9  # 波长

    I_meas_list = []
    for (i, file_path) in enumerate(file_path_list):
        data_crop, xi, xj = preprocess_data(file_path, crop_x=(165+3*i,205+3*i), crop_y=(1360+4*i,1400+4*i))
        # 将 numpy 转为 cupy
        I_meas_list.append(cp.asarray(data_crop, dtype=cp.float64))

    # Upsampling
    I_meas_list = upsample_meas_list(I_meas_list, up=4)
    dx = dx / 4  # 更新采样间距

    # plt.imshow(cp.asnumpy(I_meas_list[0]), cmap='inferno')
    # plt.show()
    
    # z_list 可以根据你的测量 z 位置设置（这里用 mm -> m 转换）
    # 假设 file_path_list 按 z 顺序排列
    z_positions_mm = np.arange(10.5, 15.0, 1)  # 示例
    z_list = (z_positions_mm - z_positions_mm[0]) * 1.0e-3 * 1  # 相对第一个平面
    print(z_positions_mm)
    
    # 波前重建
    E_rec, residuals = multi_plane_gs(
        I_meas_list,
        z_list,
        dx,
        wavelength,
        n_iter=400,
        relax=0.,
        init_field=cp.sqrt(cp.abs(I_meas_list[0])),
        smooth_phase_sigma=0,
        # energy_normalize=True,
        verbose=True
    )

    # E_rec = I_meas_list[0]**0.5 * cp.exp(1j*cp.angle(E_rec))  # 用测量振幅重新调制近场

    # 可视化结果

    # z_evol_list = np.arange(10.5, 15.0, 1)

    compare_intensity_evolution(
        reconstructed_field=E_rec.copy(),
        I_meas_list=I_meas_list,
        z_list=z_list,
        wavelength=wavelength,
        dx=dx
    )

    visualize_reconstruction(E_rec.copy())

    # 残差曲线
    # plt.figure()
    # plt.plot(residuals)
    # plt.yscale('log')
    # plt.xlabel("Iteration")
    # plt.ylabel("Mean relative error")
    # plt.title("Multi-plane GS Residuals")
    # plt.show()

    Ez, _ = angular_spectrum_propagation(E_rec, dx, wavelength, -1e-3, pad_factor=1)
    I_prop = cp.abs(Ez)**2
    I_prop= I_prop.get()

    plt.imshow(I_prop, cmap='inferno', origin='lower')
    plt.title("Propagated Intensity @ z=-1mm")
    plt.colorbar()
    plt.savefig("./-1.jpg")
    plt.close()

    Ez, _ = angular_spectrum_propagation(E_rec, dx, wavelength, 1e-3, pad_factor=1)
    I_prop = cp.abs(Ez)**2
    I_prop= I_prop.get()

    plt.imshow(I_prop, cmap='inferno', origin='lower')
    plt.title("Propagated Intensity @ z=1mm")
    plt.colorbar()
    plt.savefig("./1.jpg")
    plt.close()
