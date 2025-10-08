import cupy as cp

# import cupyx
from cupyx.scipy.ndimage import gaussian_filter
from cupy.fft import fft2, ifft2, fftfreq

# import matplotlib.pyplot as plt  # only for example plotting at the end (use cp.asnumpy)


# ---------------------------
# Multi-plane Gerchberg-Saxton (CuPy)
# ---------------------------
def multi_plane_gs(
    I_meas_list,
    z_list,
    dx,
    wavelength,
    n_iter=200,
    init_field=None,
    support_mask=None,
    relax=0.0,
    smooth_phase_sigma=None,
    verbose=False,
):
    """
    CuPy GPU implementation of multi-plane Gerchberg-Saxton.
    Inputs:
      I_meas_list : list of 2D arrays (numpy or cupy) measured intensities (same shape)
      z_list      : list of distances (m) relative to reconstruction plane
      dx, wavelength : pixel size and wavelength
      n_iter      : number of iterations
      init_field  : optional initial complex field at recon plane (numpy or cupy)
      support_mask: optional boolean mask (numpy or cupy) applied at recon plane
      relax       : relaxation blending (0 -> classic GS)
      smooth_phase_sigma : if not None, gaussian sigma in pixels to smooth phase each iter
    Returns:
      E (cupy array): reconstructed complex field at reconstruction plane (on GPU)
      residuals (list of floats): relative residual per iteration
    """
    # move measurements to GPU arrays
    I_meas_list = [cp.asarray(im) for im in I_meas_list]
    Ny, Nx = I_meas_list[0].shape
    assert all(
        im.shape == (Ny, Nx) for im in I_meas_list
    ), "All I_meas must share same shape"
    nplanes = len(I_meas_list)

    amp_meas_list = [cp.sqrt(cp.maximum(im, 0)) for im in I_meas_list]

    # initial field (on GPU)
    if init_field is None:
        init_amp = cp.mean(cp.stack(amp_meas_list, axis=0), axis=0)
        rs = cp.random.RandomState(0)
        init_phase = rs.uniform(-cp.pi, cp.pi, size=(Ny, Nx))
        E = init_amp * cp.exp(1j * init_phase)
    else:
        E = cp.asarray(init_field).astype(cp.complex128)

    if support_mask is not None:
        support_mask = cp.asarray(support_mask)

    residuals = []

    for it in range(n_iter):
        E_prev = E.copy()
        # traverse measurement planes
        for amp_meas, z in zip(amp_meas_list, z_list):
            Ez = angular_spectrum_propagation(E, dx, wavelength, z)
            phi = cp.angle(Ez)
            Ez_new = amp_meas * cp.exp(1j * phi)  # replace amplitude, keep phase
            E_new = angular_spectrum_propagation(Ez_new, dx, wavelength, -z)
            if relax == 0.0:
                E = E_new
            else:
                E = (1 - relax) * E_new + relax * E_prev

        # support constraint
        if support_mask is not None:
            E = E * support_mask

        # optional phase smoothing (via gaussian on phase)
        if smooth_phase_sigma is not None:
            amp = cp.abs(E)
            phase = cp.angle(E)
            # gaussian_filter from cupyx.scipy.ndimage works on GPU arrays
            phase_s = gaussian_filter(phase, sigma=smooth_phase_sigma)
            E = amp * cp.exp(1j * phase_s)

        # compute residual (mean relative error across planes)
        total_err = 0.0
        for amp_meas, z in zip(amp_meas_list, z_list):
            Ez = angular_spectrum_propagation(E, dx, wavelength, z)
            total_err += cp.linalg.norm(cp.abs(Ez) - amp_meas) / (
                cp.linalg.norm(amp_meas) + 1e-16
            )
        residuals.append((total_err / nplanes).item())  # .item() -> python float

        if verbose and (it % max(1, n_iter // 10) == 0 or it == n_iter - 1):
            print(f"iter {it+1}/{n_iter}, mean rel err: {residuals[-1]:.4e}")

    return E, residuals
