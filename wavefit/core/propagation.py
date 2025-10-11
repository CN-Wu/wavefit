import cupy as cp
import numpy as np

def _next_pow2(x):
    return 1 << (int(np.ceil(np.log2(x))))

def angular_spectrum_propagation(E0, dx, wavelength, z,
                                 pad_factor=1, dtype=cp.complex128, safe=True):
    """
    单个距离 z 的 Angular Spectrum (ASM) 传播（GPU，CuPy）。
    - E0: complex 2D cupy array (ny, nx)
    - dx: pixel spacing (m)
    - wavelength: (m)
    - z: propagation distance (m) (can be negative)
    - pad_factor: integer >=1, 如果 >1 则在每个方向上 pad 到 pad_factor * size (并向上取 2 的幂)
    返回：同尺寸的 complex cupy array（裁剪回原始 ny,nx）
    关键点：不使用 fftshift，使用 cp.fft.fftfreq 直接构造频域核 H
    """
    E0 = cp.asarray(E0, dtype=dtype)
    ny, nx = E0.shape

    # === padding ===
    if pad_factor is None or pad_factor <= 1:
        Ny = ny; Nx = nx
        Epad = E0
        y0 = x0 = 0
    else:
        Nx_req = int(np.ceil(nx * pad_factor))
        Ny_req = int(np.ceil(ny * pad_factor))
        Nx = _next_pow2(Nx_req)
        Ny = _next_pow2(Ny_req)
        pad_x1 = (Nx - nx)//2
        pad_x2 = Nx - nx - pad_x1
        pad_y1 = (Ny - ny)//2
        pad_y2 = Ny - ny - pad_y1
        Epad = cp.pad(E0, ((pad_y1, pad_y2), (pad_x1, pad_x2)))
        y0, x0 = pad_y1, pad_x1

    # frequency grids (cycles per meter)
    fx = cp.fft.fftfreq(Epad.shape[1], d=dx)  # length Nx
    fy = cp.fft.fftfreq(Epad.shape[0], d=dx)  # length Ny
    FX, FY = cp.meshgrid(fx, fy)  # shape (Ny, Nx), matches FFT ordering

    # wave numbers
    k = 2 * cp.pi / wavelength
    kz_sq = (k ** 2) - (2 * cp.pi * FX) ** 2 - (2 * cp.pi * FY) ** 2
    if safe:
        kz_sq = cp.where(kz_sq < 0, 0, kz_sq)
    kz = cp.sqrt(kz_sq.astype(cp.complex128))

    # # complex sqrt to keep evanescent components
    # kz = 2 * cp.pi * cp.sqrt(H_arg.astype(cp.complex128))

    # propagation kernel
    H = cp.exp(1j * kz * z)

    # forward FFT, multiply, inverse FFT
    Epad_f = cp.fft.fft2(Epad)
    Ez_pad = cp.fft.ifft2(Epad_f * H)

    # crop back to original size
    if pad_factor is None or pad_factor <= 1:
        return Ez_pad, None
    else:
        return Ez_pad[y0:y0+ny, x0:x0+nx], Ez_pad

def angular_spectrum_propagation_batch(E0, dx, wavelength, z_list, pad_factor=1, dtype=cp.complex128, safe=True):
    """
    Batch 版本：给定一个 E0，返回对多个 z 的传播结果 list（尽量重用 FFT）。
    """
    E0 = cp.asarray(E0, dtype=dtype)
    ny, nx = E0.shape

    # padding as above
    if pad_factor is None or pad_factor <= 1:
        Ny = ny; Nx = nx
        Epad = E0
        y0 = x0 = 0
    else:
        Nx_req = int(np.ceil(nx * pad_factor))
        Ny_req = int(np.ceil(ny * pad_factor))
        Nx = _next_pow2(Nx_req)
        Ny = _next_pow2(Ny_req)
        pad_x1 = (Nx - nx)//2; pad_x2 = Nx - nx - pad_x1
        pad_y1 = (Ny - ny)//2; pad_y2 = Ny - ny - pad_y1
        Epad = cp.pad(E0, ((pad_y1, pad_y2), (pad_x1, pad_x2)))
        y0, x0 = pad_y1, pad_x1

    fx = cp.fft.fftfreq(Epad.shape[1], d=dx)
    fy = cp.fft.fftfreq(Epad.shape[0], d=dx)
    FX, FY = cp.meshgrid(fx, fy)
    k = 2 * cp.pi / wavelength
    H_arg = (1.0 / (wavelength**2)) - (FX**2 + FY**2)
    if safe:
        H_arg = cp.where(H_arg < 0, 0, H_arg)  # 避免负数开根号

    kz_base = 2 * cp.pi * cp.sqrt(H_arg.astype(cp.complex128))

    Epad_f = cp.fft.fft2(Epad)
    outs = []
    for z in z_list:
        H = cp.exp(1j * kz_base * z)
        Ez_pad = cp.fft.ifft2(Epad_f * H)
        if pad_factor is None or pad_factor <= 1:
            outs.append(Ez_pad)
        else:
            outs.append(Ez_pad[y0:y0+ny, x0:x0+nx])
    return outs
