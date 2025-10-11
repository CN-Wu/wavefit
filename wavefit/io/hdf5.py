import h5py

def read_hdf5_dataset(file_path, dataset_name):
    """
    Reads a dataset from an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to read.

    Returns:
        numpy.ndarray: The data from the specified dataset.
    """
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name][:]
    return data

def list_hdf5_datasets(file_path):
    """
    Lists all datasets in an HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.

    Returns:
        list: List of dataset names.
    """
    datasets = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append(name)
    with h5py.File(file_path, 'r') as f:
        f.visititems(visitor)
    return datasets


import h5py
import numpy as np
import matplotlib.pyplot as plt

def restore_physical_intensity(h5_path, frame_index=1, show_plot=True):
    """
    从 BeamGage / LBP2 HDF5 文件恢复物理强度 (W/m^2)。
    优先策略：
      1) 若存在 ENERGY/ENERGYOFFRAME (J) 且状态有效 -> 用 total energy 标定像素比例 -> 得到每像素能量 J -> 除以曝光时间(s) 和像素面积(m^2) 得到 W/m^2
      2) 否则若存在 POWER_CALIBRATION_MULTIPLIER -> 假设其单位为 W / (unit_of_linear_count) -> 直接得到像素功率 -> 除像素面积得到 W/m^2
      3) 否则仅返回"线性计数/像素面积"（单位: arbitrary counts / m^2），并提示需要外部标定。
    返回值： dict 包含 keys:
      - intensity_W_m2 : 2D numpy array (W/m^2) 或 None
      - linear_counts  : 2D numpy array (线性计数)
      - pixel_area_m2  : float
      - meta           : dict (包含用于计算的原始元数据)
    """
    meta = {}
    with h5py.File(h5_path, 'r') as f:
        base = f[f'BG_DATA/{frame_index}']

        # 读取原始 DATA（可能为整数）
        data_raw = np.array(base['DATA']).astype(np.float64)
        # --- reshape ---
        height = base['RAWFRAME/HEIGHT'][()][0]
        width  = base['RAWFRAME/WIDTH'][()][0]
        data_raw = data_raw.reshape((height, width))

        # read camera params if present
        def get(path, default=None):
            try:
                return base[path][()][0]
            except Exception:
                return default

        # keys from your list
        black = get('RAWFRAME/BLACKLEVELSTAMP', 0.0)
        gain = get('RAWFRAME/GAINSTAMP', 1.0)
        gamma = get('RAWFRAME/ORIGINALGAMMA', 1.0)
        gam_corr = get('GAMMACORRECTED', False)
        sub_off = get('RAWFRAME/SUBTRACTION_OFFSET', 0.0)
        px_x_um = get('RAWFRAME/PIXELSCALEXUM', None)
        px_y_um = get('RAWFRAME/PIXELSCALEYUM', None)
        exposure_stamp = get('RAWFRAME/EXPOSURESTAMP', None)  # ms? BeamGage often in ms
        # energy fields
        energy_of_frame = get('RAWFRAME/ENERGY/ENERGYOFFRAME', None)
        energy_status = get('RAWFRAME/ENERGY/ENERGYOFBEAMSTATUS', None)
        power_cal_mult = get('RAWFRAME/ENERGY/POWER_CALIBRATION_MULTIPLIER', None)
        print(black, gain, gamma, gam_corr, sub_off, px_x_um, px_y_um, exposure_stamp, energy_of_frame, energy_status, power_cal_mult)

        # pack meta
        meta.update({
            'blacklevel': float(black) if black is not None else None,
            'gain': float(gain) if gain is not None else None,
            'gamma': float(gamma) if gamma is not None else None,
            'gam_corrected_flag': bool(gam_corr) if gam_corr is not None else None,
            'subtraction_offset': float(sub_off) if sub_off is not None else None,
            'pixel_scale_x_um': float(px_x_um) if px_x_um is not None else None,
            'pixel_scale_y_um': float(px_y_um) if px_y_um is not None else None,
            'exposure_stamp': float(exposure_stamp) if exposure_stamp is not None else None,
            'energy_of_frame': float(energy_of_frame) if energy_of_frame is not None else None,
            'power_calibration_multiplier': float(power_cal_mult) if power_cal_mult is not None else None,
        })
        # energy_status 处理
        if energy_status is None:
            energy_status_val = None
        elif isinstance(energy_status, (bytes, str)):
            s = energy_status.decode() if isinstance(energy_status, bytes) else energy_status
            s = s.strip().upper()
            if s in ('CALIBRATED', 'VALID', '1'):
                energy_status_val = 1
            elif s in ('NOT_CALIBRATED', 'INVALID', '0'):
                energy_status_val = 0
            else:
                # 尝试数字
                try:
                    energy_status_val = int(s)
                except:
                    energy_status_val = None
        else:
            # 数值类型直接用
            try:
                energy_status_val = int(energy_status)
            except:
                energy_status_val = None

        meta['energy_status'] = energy_status_val

        print(meta)


    # --- STEP 1: 线性化 / 还原计数 ---
    # 先移除 offset / blacklevel
    linear = data_raw.astype(np.float64)
    # subtract subtraction offset first if present
    if meta['subtraction_offset'] is not None:
        linear = linear - meta['subtraction_offset']
    # subtract blacklevel
    if meta['blacklevel'] is not None:
        linear = linear - meta['blacklevel']

    # 如果数据经过 gamma 校正（或文件里标明 GAMMACORRECTED=True），尝试反伽马
    if meta['gam_corrected_flag'] and meta['gamma'] and meta['gamma'] != 1.0:
        # 归一化到最大(避免负值)
        mx = np.nanmax(linear)
        if mx <= 0:
            print("Warning: after subtracting offsets, data max <= 0; can't inverse gamma reliably.")
            # fallback: don't inverse gamma
        else:
            lin_norm = linear / mx
            # clip numerically
            lin_norm = np.clip(lin_norm, 0.0, None)
            inv = np.power(lin_norm, 1.0 / float(meta['gamma']))
            # scale back by mx (we assume gamma operated on normalized values)
            linear = inv * mx
    # 若没有 gamma 或 gamma==1，则跳过

    # 如果 gain 字段意义是“将计数换算成某种线性单位”的倍乘/除，需要根据实际文件解释处理。
    # 常见做法：如果 gain 是放大倍数，则除以 gain（或乘以，视实现而定）。
    # 这里无法一概而论，我们先将 gain 作为“counts -> physical count units 的乘数”：
    if meta['gain'] is not None and meta['gain'] != 0:
        # 我们假定 "linear_counts = linear * gain"（如果你发现结果太大/太小，可以改为除）
        linear_counts = linear * float(meta['gain'])
    else:
        linear_counts = linear.copy()

    # 保证非负
    linear_counts = np.clip(linear_counts, 0.0, None)

    # --- STEP 2: 像素面积 & 曝光时间 ---
    # 像素尺寸：优先使用 px_x_um/px_y_um；如果缺失，尝试从 RAWFRAME/WIDTH/HEIGHT/MAXWIDTH/MAXHEIGHT 不能推单位，必须有像素尺寸
    if meta['pixel_scale_x_um'] is None or meta['pixel_scale_y_um'] is None:
        print("Warning: pixel scale (µm) missing in file; cannot convert to physical area. Returning counts per pixel only.")
        pixel_area_m2 = None
    else:
        px_x_m = meta['pixel_scale_x_um'] * 1e-6
        px_y_m = meta['pixel_scale_y_um'] * 1e-6
        pixel_area_m2 = float(px_x_m * px_y_m)

    # 曝光时间：BeamGage 的 EXPOSURESTAMP 有时以 ms 为单位，若显然过大（>10 s）或为 None，请自行确认单位
    exposure_s = None
    if meta['exposure_stamp'] is not None:
        # 尝试判断单位：若值 > 1e3，很可能是 microseconds or nanoseconds? 常见是 ms
        # 大多数 BeamGage 是以 ms 保存，这里我们假设为 ms 并转换
        exposure_s = float(meta['exposure_stamp']) / 1000.0  # ms -> s
    # 如果 exposure_s 为空则后续无法算出功率（W），只能给能量或 counts。

    # STEP 3: 标定：优先使用 ENERGYOFFRAME/POWER_CALIBRATION_MULTIPLIER
    intensity_W_m2 = None
    method_used = None

    energy_ok = meta['energy_of_frame'] is not None and meta['energy_of_frame'] > 0 and meta['energy_status'] == 1
    power_mult_ok = meta['power_calibration_multiplier'] is not None and meta['power_calibration_multiplier'] > 0

    if energy_ok:
        # 能量标定可用
        total_counts = np.sum(linear_counts)
        if total_counts > 0 and exposure_s is not None and pixel_area_m2 is not None:
            pixel_energy_J = linear_counts * (meta['energy_of_frame'] / total_counts)
            pixel_power_W = pixel_energy_J / exposure_s
            intensity_W_m2 = pixel_power_W / pixel_area_m2
            method_used = 'energy_of_frame'
    elif power_mult_ok:
        # POWER_CALIBRATION_MULTIPLIER 可用
        pixel_power_W = linear_counts * meta['power_calibration_multiplier']
        if pixel_area_m2 is not None:
            intensity_W_m2 = pixel_power_W / pixel_area_m2
        else:
            intensity_W_m2 = pixel_power_W
        method_used = 'power_calibration_multiplier'
    else:
        # 标定不可用 → 仅返回线性计数
        if pixel_area_m2 is not None:
            intensity_W_m2 = linear_counts / pixel_area_m2
            method_used = 'counts_per_m2_no_cal'
        else:
            intensity_W_m2 = linear_counts.copy()
            method_used = 'counts_no_cal'

    result = {
        'intensity_W_m2': intensity_W_m2,
        'linear_counts': linear_counts,
        'pixel_area_m2': pixel_area_m2,
        'exposure_s': exposure_s,
        'method_used': method_used,
        'meta': meta
    }

    # optional plotting
    if show_plot:
        plt.figure(figsize=(6,5))
        if intensity_W_m2 is not None:
            im = plt.imshow(intensity_W_m2, origin='lower')
            plt.title(f"Intensity ({method_used})")
            cbar = plt.colorbar(im)
            label = "W/m²" if method_used in ('energy_of_frame','power_calibration_multiplier') else "counts / m²"
            cbar.set_label(label)
        else:
            plt.imshow(linear_counts, origin='lower')
            plt.title("Linear counts (no physical calibration)")
            plt.colorbar(label='counts')
        plt.xlabel("pixel x")
        plt.ylabel("pixel y")
        plt.tight_layout()
        plt.show()

    return result

# === 使用示例 ===
if __name__ == "__main__":
    path = "../../data/1030nm_Lens40cm_14.5.lbp2Data"
    res = restore_physical_intensity(path, frame_index=1, show_plot=True)
    # 输出关键结果
    print("方法:", res['method_used'])
    print("像素面积 (m^2):", res['pixel_area_m2'])
    print("曝光时间 (s):", res['exposure_s'])
    if res['intensity_W_m2'] is not None:
        arr = res['intensity_W_m2']
        print("Intensity range:", np.nanmin(arr), np.nanmax(arr))
    else:
        print("未能生成物理单位强度，只返回 linear counts。")
