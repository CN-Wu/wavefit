import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from wavefit.io.hdf5 import read_hdf5_dataset

src_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(src_path, "../../hartmann")))
try:
    from hartmann.centroid.detection import detect_centroids_from_image
except ImportError:
    raise

plt.style.use("../waveoptics/waveoptics/utils/styles/sci.mplstyle")

def dataloader(file_path, dataset_name):
    """
    Loads data from an HDF5 file and returns it as a numpy array.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_name (str): Name of the dataset to load.

    Returns:
        numpy.ndarray: The data from the specified dataset.
    """
    data = read_hdf5_dataset(file_path, dataset_name).reshape([1448, 1928])
    return data


def visualize_data(file_path):
    """
    Visualizes the data using matplotlib.

    Args:
        data (numpy.ndarray): The data to visualize.
    """
    dataset_name = "BG_DATA/1/DATA"

    # Load data
    data = dataloader(file_path, dataset_name)

    # Print data shape and type
    print("Data shape:", data.shape)
    print("Data type:", data.dtype)

    # # Visualize the data
    # plt.imshow(data, cmap='gray', origin='lower')
    # plt.colorbar()
    # plt.title('USST Data Visualization')
    # plt.show()

    # Data truncation
    range_xi = np.array([160, 220])
    range_xj = np.array([1355, 1415])
    data = data[range_xi[0]:range_xi[1], range_xj[0]:range_xj[1]]
    print("Truncated data shape:", data.shape)

    centroid, _, _ =  detect_centroids_from_image(data)
    centroid_xi, centroid_xj = centroid[0]
    print("Centroid position (xi, xj):", centroid_xi, centroid_xj)

    # Visualize the truncated data
    plt.imshow(data, cmap='hot', origin='lower')
    plt.colorbar()
    plt.scatter(centroid_xj, centroid_xi, color='cyan', marker="x", s=8)
    plt.title(f"{file_path.split('1030nm_Lens40cm_')[-1].split('.lbp2Data')[0]}")
    plt.savefig(f"{file_path.split('1030nm_Lens40cm_')[-1].split('.lbp2Data')[0]}_truncated.png")
    plt.close()

    return centroid_xi, centroid_xj


if __name__ == "__main__":
    file_path_list = [f"./data/1030nm_Lens40cm_{i:.1f}.lbp2Data" for i in np.arange(10.5, 15.5)]
    centroid_xi_list = np.zeros_like(file_path_list, dtype=float)
    centroid_xj_list = np.zeros_like(file_path_list, dtype=float)
    for i, file_path in enumerate(file_path_list):
        print(f"Processing file: {file_path}")
        centroid_xi, centroid_xj = visualize_data(file_path)
        centroid_xi_list[i] = centroid_xi
        centroid_xj_list[i] = centroid_xj

    plt.plot(np.arange(10.5, 15.5), centroid_xj_list - centroid_xj_list[0], marker='o')
    plt.plot(np.arange(10.5, 15.5), centroid_xi_list - centroid_xi_list[0], marker='o')
    plt.xlabel('z (mm)')
    plt.ylabel('Centroid Position (pixels)')
    plt.legend(['Centroid Xj', 'Centroid Xi'])
    plt.grid()
    plt.show()

    # Total shift is 15*17 pixels over 4 mm, so the angle between k and the optical axis of CCD is about (16 * 3.69 um) / 4 mm = 0.015, negligible
    