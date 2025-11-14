import cupy as cp

# import cupyx
from cupyx.scipy.ndimage import gaussian_filter
from cupy.fft import fft2, ifft2, fftfreq
from wavefit.core.propagation import angular_spectrum_propagation

# import matplotlib.pyplot as plt  # only for example plotting at the end (use cp.asnumpy)

import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter

def multi_plane_gs(
    I_meas_list,
    z_list,
    dx,
    wavelength,
    n_iter=200,
    init_field=None,
    support_mask=None,
    relax=0.0,
    relax_mode='blend',          # 'blend' (E = (1-r)E_new + r E_prev), 'incr' (E = E_prev + r*(E_new - E_prev)), 'hio' (simple HIO-like)
    relax_schedule=None,         # list/array of length <= n_iter to override relax per iteration
    smooth_phase_sigma=None,     # sigma in pixels for gaussian smoothing of the complex phasor
    tol=None,                    # optional stopping tol on residual
    patience=10,                 # for early stopping if tol provided
    energy_normalize=True,       # normalize energy after update
    verbose=False,
):
    """
    Improved multi-plane Gerchberg-Saxton on GPU (CuPy).
    Required:
      - I_meas_list: list of 2D arrays (numpy or cupy) measured intensities (same shape)
      - z_list: list of propagation distances (m), same length as I_meas_list
      - dx, wavelength: pixel size and wavelength
      - angular_spectrum_propagation: callable for forward/back propagation

    Returns:
      E (cupy array): reconstructed complex field at reconstruction plane
      residuals (list of floats)
    """
    # convert measurements to GPU arrays and amplitudes
    I_meas_list = [cp.asarray(im).astype(cp.float64) for im in I_meas_list]
    Ny, Nx = I_meas_list[0].shape
    assert all(im.shape == (Ny, Nx) for im in I_meas_list), "All I_meas must share same shape"

    amp_meas_list = [cp.sqrt(cp.maximum(im, 0)) for im in I_meas_list]
    nplanes = len(amp_meas_list)

    # init field
    if init_field is None:
        # use mean amplitude and random phase
        init_amp = cp.mean(cp.stack(amp_meas_list, axis=0), axis=0)
        rs = cp.random.RandomState(2)
        init_phase = rs.uniform(-cp.pi, cp.pi, size=(Ny, Nx))
        E = init_amp * cp.exp(1j * init_phase)
    else:
        E = cp.asarray(init_field).astype(cp.complex128)

    if support_mask is not None:
        support_mask = cp.asarray(support_mask).astype(cp.float64)

    # bookkeeping
    residuals = []
    best_res = cp.inf
    no_improve = 0

    # precompute initial energy
    def energy(arr):
        return cp.sum(cp.abs(arr) ** 2)

    E_energy_ref = energy(E) + 1e-16

    # iterator over relax schedule
    if relax_schedule is None:
        def get_relax(it):
            return relax
    else:
        rsched = cp.asarray(relax_schedule)
        def get_relax(it):
            if it < len(rsched):
                return float(rsched[it])
            return float(rsched[-1])

    for it in range(n_iter):
        E_prev = E.copy()

        # forward pass: compute Ez for each plane and replace amplitude
        Ez_cache = []
        # i=0
        for amp_meas, z in zip(amp_meas_list, z_list):
            # print(i)
            # i += 1
            Ez, _ = angular_spectrum_propagation(E, dx, wavelength, z, pad_factor=1)
            Ez_cache.append(Ez)
            # replace amplitude, keep measured amplitude and phase of Ez
            phi = cp.angle(Ez)
            Ez_new = amp_meas * cp.exp(1j * phi)
            # inverse propagate back
            E_new, _ = angular_spectrum_propagation(Ez_new, dx, wavelength, -z, pad_factor=1)

            # update rule according to relax_mode and relax value
            r = get_relax(it)
            if relax_mode == 'blend':
                # classic blending between new estimate and previous estimate
                E = (1 - r) * E_new + r * E_prev
            elif relax_mode == 'incr':
                # incremental update: E = E_prev + r * (E_new - E_prev)
                E = E_prev + r * (E_new - E_prev)
            elif relax_mode == 'hio' and support_mask is not None:
                # simple HIO-like: inside support use Ez_new, outside use prev - beta * Ez_new
                # note: HIO normally applies in object plane. Here we apply a simple variant.
                inside = support_mask.astype(bool)
                beta = r
                E_candidate = E_new
                E = cp.where(inside, E_candidate, E_prev - beta * E_candidate)
            else:
                # fallback
                E = E_new

        # support constraint (enforce magnitude/phase within mask)
        if support_mask is not None:
            E = E * support_mask

        # optional phase smoothing via complex phasor filtering (avoid wrap issues)
        if smooth_phase_sigma is not None:
            amp = cp.abs(E)
            phase = cp.angle(E)
            phasor = cp.exp(1j * phase)
            # apply gaussian_filter to real and imag separately
            real_s = gaussian_filter(cp.real(phasor), sigma=smooth_phase_sigma)
            imag_s = gaussian_filter(cp.imag(phasor), sigma=smooth_phase_sigma)
            phasor_s = real_s + 1j * imag_s
            # renormalize phasor magnitude (to 1) then reconstruct field
            phasor_s = phasor_s / (cp.abs(phasor_s) + 1e-16)
            E = amp * phasor_s

        # energy normalization to match reference (prevents drift)
        if energy_normalize:
            curr_energy = energy(E) + 1e-16
            E *= cp.sqrt(E_energy_ref / curr_energy)

        # compute residuals using cached Ez (faster)
        total_err = 0.0
        for amp_meas, z, Ez_forward in zip(amp_meas_list, z_list, Ez_cache):
            # Ez_forward = angular_spectrum_propagation(E_prev, dx, wavelength, z)  # already computed for E_prev during loop
            # BUT Ez_forward currently corresponds to E before replacement in that loop iteration; we need Ez from current E
            # So compute Ez from current E (only one propagation per plane, unavoidable for residual correctness)
            Ez_now, _ = angular_spectrum_propagation(E, dx, wavelength, z)
            total_err += cp.linalg.norm(cp.abs(Ez_now) - amp_meas) / (cp.linalg.norm(amp_meas) + 1e-16)

        mean_rel_err = (total_err / nplanes).item()
        residuals.append(mean_rel_err)

        # verbose
        if verbose and (it % max(1, n_iter // 10) == 0 or it == n_iter - 1):
            print(f"iter {it+1}/{n_iter}, mean rel err: {mean_rel_err:.4e}, relax={get_relax(it):.3f}")

        # early stopping logic
        if tol is not None:
            if mean_rel_err + 1e-12 < best_res:
                best_res = mean_rel_err
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at iter {it+1} (no improvement for {patience} iters).")
                break

    return E, residuals

# ---------------------------
# Multi-plane Gerchberg-Saxton (CuPy)
# ---------------------------
# def multi_plane_gs(
#     I_meas_list,
#     z_list,
#     dx,
#     wavelength,
#     n_iter=200,
#     init_field=None,
#     support_mask=None,
#     relax=0.0,
#     smooth_phase_sigma=None,
#     verbose=False,
# ):
#     """
#     CuPy GPU implementation of multi-plane Gerchberg-Saxton.
#     Inputs:
#       I_meas_list : list of 2D arrays (numpy or cupy) measured intensities (same shape)
#       z_list      : list of distances (m) relative to reconstruction plane
#       dx, wavelength : pixel size and wavelength
#       n_iter      : number of iterations
#       init_field  : optional initial complex field at recon plane (numpy or cupy)
#       support_mask: optional boolean mask (numpy or cupy) applied at recon plane
#       relax       : relaxation blending (0 -> classic GS)
#       smooth_phase_sigma : if not None, gaussian sigma in pixels to smooth phase each iter
#     Returns:
#       E (cupy array): reconstructed complex field at reconstruction plane (on GPU)
#       residuals (list of floats): relative residual per iteration
#     """
#     # move measurements to GPU arrays
#     I_meas_list = [cp.asarray(im) for im in I_meas_list]
#     Ny, Nx = I_meas_list[0].shape
#     assert all(
#         im.shape == (Ny, Nx) for im in I_meas_list
#     ), "All I_meas must share same shape"
#     nplanes = len(I_meas_list)

#     amp_meas_list = [cp.sqrt(cp.maximum(im, 0)) for im in I_meas_list]

#     # initial field (on GPU)
#     if init_field is None:
#         init_amp = cp.mean(cp.stack(amp_meas_list, axis=0), axis=0)
#         rs = cp.random.RandomState(0)
#         init_phase = rs.uniform(-cp.pi, cp.pi, size=(Ny, Nx)) * 0
#         E = init_amp * cp.exp(1j * init_phase)
#     else:
#         E = cp.asarray(init_field).astype(cp.complex128)

#     if support_mask is not None:
#         support_mask = cp.asarray(support_mask)

#     residuals = []

#     for it in range(n_iter):
#         E_prev = E.copy()
#         # traverse measurement planes
#         for amp_meas, z in zip(amp_meas_list, z_list):
#             Ez = angular_spectrum_propagation(E, dx, wavelength, z)
#             phi = cp.angle(Ez)
#             Ez_new = amp_meas * cp.exp(1j * phi)  # replace amplitude, keep phase
#             E_new = angular_spectrum_propagation(Ez_new, dx, wavelength, -z)
#             if relax == 0.0:
#                 E = E_new
#             else:
#                 E = (1 - relax) * E_new + relax * E_prev

#         # support constraint
#         if support_mask is not None:
#             E = E * support_mask

#         # optional phase smoothing (via gaussian on phase)
#         if smooth_phase_sigma is not None:
#             amp = cp.abs(E)
#             phase = cp.angle(E)
#             # gaussian_filter from cupyx.scipy.ndimage works on GPU arrays
#             phase_s = gaussian_filter(phase, sigma=smooth_phase_sigma)
#             E = amp * cp.exp(1j * phase_s)

#         # compute residual (mean relative error across planes)
#         total_err = 0.0
#         for amp_meas, z in zip(amp_meas_list, z_list):
#             Ez = angular_spectrum_propagation(E, dx, wavelength, z)
#             total_err += cp.linalg.norm(cp.abs(Ez) - amp_meas) / (
#                 cp.linalg.norm(amp_meas) + 1e-16
#             )
#         residuals.append((total_err / nplanes).item())  # .item() -> python float

#         if verbose and (it % max(1, n_iter // 10) == 0 or it == n_iter - 1):
#             print(f"iter {it+1}/{n_iter}, mean rel err: {residuals[-1]:.4e}")

#     return E, residuals


def gerchberg_saxton(I_near, I_far, dx, wavelength, z,
                     pad_factor=2, n_iter=50, verbose=True):
    """
    GPU Gerchberg–Saxton phase retrieval
    输入：
      I_near: 近场强度 (cupy array)
      I_far: 远场强度 (cupy array)
      dx: 采样间隔 (m)
      wavelength: 波长 (m)
      z: 传播距离 (m)
    返回：
      近场估计的复场 (cupy array)
    """
    # 初始化随机相位
    rng = cp.random.RandomState(42)
    phase = rng.uniform(-cp.pi, cp.pi, I_near.shape)*0
    E = cp.sqrt(I_near) * cp.exp(1j * phase)

    for i in range(n_iter):
        # Forward propagation
        Ez = angular_spectrum_propagation(E, dx, wavelength, z, pad_factor=pad_factor)

        # 替换振幅为测量强度的平方根
        Ez = cp.sqrt(I_far) * cp.exp(1j * cp.angle(Ez))

        # Backward propagation
        E_back = angular_spectrum_propagation(Ez, dx, wavelength, -z, pad_factor=pad_factor)

        # 替换振幅为近场强度平方根
        E = cp.sqrt(I_near) * cp.exp(1j * cp.angle(E_back))

        if verbose and i % 10 == 0:
            mse = cp.mean(cp.abs(E_back - E)**2)
            print(f"Iteration {i:03d} | MSE={float(mse):.4e}")

    return E