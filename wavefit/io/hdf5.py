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