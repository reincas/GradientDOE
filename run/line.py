##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import h5py
import matplotlib.pyplot as plt


def plot_hdf5_row(file_path, dataset_name, row_index, slice):
    try:
        # Open the HDF5 file in read-only mode
        with h5py.File(file_path, 'r') as h5_file:
            # Access the dataset
            dataset = h5_file[dataset_name]

            # Check if the dataset is 2D
            if len(dataset.shape) != 2:
                print(f"Error: Dataset '{dataset_name}' is {len(dataset.shape)}D, but a 2D array is required.")
                return

            # Extract the specific row
            # Using slicing [row_index, :] is efficient as it only loads that row into memory
            row_data = dataset[row_index, slice]

            # Plotting the data
            plt.figure(figsize=(10, 5))
            plt.plot(row_data, marker='o', linestyle='-', markersize=2)
            plt.title(f"Dataset: {dataset_name} | Row Index: {row_index}")
            plt.xlabel("Column Index")
            plt.ylabel("Value")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except KeyError:
        print(f"Error: Dataset '{dataset_name}' not found in the file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    path = "./result_07/height.h5"
    name = "height_1024"
    plot_hdf5_row(path, name, 500, slice(400, 600))