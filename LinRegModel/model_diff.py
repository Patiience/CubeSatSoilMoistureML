# Import other necessary libraries
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Define function to retrieve the data from the SMNDVI binaries
def ret_smndvi_bin(directory_path):
    # List used for testing later
    file_list = []

    # Sort directory to ensure we get correct order of files
    directory = os.listdir(directory_path)
    sorted_directory = sorted(directory)

    # Loop through files in directory
    for files in sorted_directory:
        # File path
        file_path = os.path.join(directory_path, files)

        # Append file to files list for evaluation later
        file_list.append(file_path)

    # Return list of files
    return file_list


# Define function to retrieve the data from the MODIS binaries
def ret_modis_bin(directory_path):
    # Sort directory to ensure we get correct order of files
    directory = os.listdir(directory_path)

    # Only include years from 2018-2024
    included_years = {'2018', '2019', '2020', '2021', '2022', '2023', '2024'}

    # Filter the directory list to include only specified years
    filtered_directory = [item for item in directory if item in included_years]

    # Sort the remaining items
    sorted_directory = sorted(filtered_directory)

    # List used for testing later
    file_list = []

    # Loop through each year's directory
    for year in sorted_directory:
        # File path
        year_path = os.path.join(directory_path, year)

        # Sort the files in the directory for matching purposes with SMNDVI
        files = os.listdir(year_path)
        sorted_files = sorted(files)

        # For each year exclude, 12/31, since SMNDVI data does not include it
        sorted_files = sorted_files[:-1]

        # If the year is 2018, exclude all days before 2018 5th Week or 20180129
        if year == "2018":
            sorted_files = sorted_files[28:] # Note: 28th index since first index is 0
        elif year == "2024":
            # If the year is 2024, exclude all days after 20240331, SMNDVI only goes up to there, MODIS goes to 20240422
            sorted_files = sorted_files[:-22] 
    
        # Loop through the files in the year's directory
        for files in sorted_files:
            # File path
            file_path = os.path.join(year_path, files)

            # Append file path to files list for evaluation later
            file_list.append(file_path)

    # Return list of files
    return file_list
        

# Define function to perform evaluation for the Linear Regression Model
def model_metrics(slope_grid, intercept_grid, smndvi_files, modis_files):
    # Choose a random sample, as it would be too much to plot all the comparisons
    sample_size = int(0.01 * len(smndvi_files))
    indices = np.random.choice(len(smndvi_files), size=sample_size, replace=False)
    smndvi_files = np.array(smndvi_files)[indices]
    modis_files = np.array(modis_files)[indices]

    # Calculate Predicted MODIS value for each SMNDVI file, then compare to actual value of corresponding MODIS
    for i in range(len(smndvi_files)):
        # Load in file paths for SMNDVI and MODIS data
        file_smndvi = smndvi_files[i]
        file_modis = modis_files[i]

        # Get date for two files
        date = file_smndvi.split("/")[4]
        date = date.split("_")[1]
        date = date.split(".")[0]

        # Load in data from binary file for SMNDVI
        with open(file_smndvi, 'rb') as f:
            smndvi_grid = pickle.load(f)

        # Convert to numpy array & mask -999, exclude values outside of 0-1 range
        smndvi_grid = np.array(smndvi_grid, dtype=np.float32)
        smndvi_grid = np.ma.masked_equal(smndvi_grid, -999)
        smndvi_grid = np.ma.masked_outside(smndvi_grid, 0, 1)

        # Load in data from binary file for MODIS data
        modis_grid = np.fromfile(file_modis, dtype='<f4')
        modis_grid = modis_grid.reshape((15000,36000))

        # Intialize lat and lon range for the NWM domain
        # NWM domain is from 20, 60 lat & -140, -60 lon
        # Addition is to adjust lat and lon values to match MODIS dataset indices
        # -60 to 90 N for lat & -180 to 180 E lon
        lat1 = int((20 + 60) / 0.01)
        lat2 = int((60 + 60)/ 0.01)
        lon1 = int((-140 + 180) / 0.01)
        lon2 = int((-60 + 180) / 0.01)

        # Retrieve the NWM domain through indexing
        modis_grid = modis_grid[lat1: lat2, lon1: lon2]

        # Mask -9999 & exclude values outside of 0-1 range
        modis_grid = np.ma.masked_equal(modis_grid, -9999)
        modis_grid = np.ma.masked_outside(modis_grid, 0, 1)

        # Calculate predicted MODIS value for SMNDVI
        # Note: Also a matrix
        modis_pred = (slope_grid * smndvi_grid) + intercept_grid

        # Flatten the data points to be plotted
        smndvi_flat = smndvi_grid.flatten()
        modis_flat = modis_grid.flatten()
        modis_pred = modis_pred.flatten()

        # For tracking purposes
        print(f"Plotting for {date}")

        # Plot Hexbin Plots for MODIS and SMNDVI before and after Linear Regression Model
        # Plot using separate hexbin plots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        # Hexbin plot for MODIS vs SMNDVI before calibration
        hb1 = axs[0].hexbin(modis_flat, smndvi_flat, gridsize=50, cmap="Blues", mincnt=1)
        axs[0].set_xlim(0, 1)  # Set x-axis limits
        axs[0].set_ylim(0, 1)  # Set y-axis limits
        axs[0].set_xlabel("MODIS NDVI")
        axs[0].set_ylabel("SMNDVI (Before Calibration)")
        axs[0].set_title("Density Plot Before Calibration")
        cb1 = fig.colorbar(hb1, ax=axs[0], label="Density")

        # Hexbin plot for MODIS vs SMNDVI after calibration
        hb2 = axs[1].hexbin(modis_flat, modis_pred, gridsize=50, cmap="Reds", mincnt=1)
        axs[1].set_xlim(0, 1)  # Set x-axis limits
        axs[1].set_ylim(0, 1)  # Set y-axis limits
        axs[1].set_xlabel("MODIS NDVI")
        axs[1].set_ylabel("SMNDVI (After Calibration)")
        axs[1].set_title("Density Plot After Calibration")
        cb2 = fig.colorbar(hb2, ax=axs[1], label="Density")

        plt.suptitle(f"MODIS vs. SMNDVI Values Before and After Calibration ({date})")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f'/data01/dlu12/LinRegModel/Cmp_{date}.png')
        plt.close()


# Main function to train and test linear regression model
if __name__ == "__main__":
    # Directory path for SMNDVI binaries retrieved earlier & MODIS NDVI
    directory_smndvi = "/data01/dlu12/All_SMNDVI_Binaries"
    directory_modis = "/data01/jyin/MODISNDVI/Gapfilling1kmNDVI"

    # Retrieve list of files from SMNDVI & MODIS dataset
    # Lists should be in order from 2018-2024 and indices should correspond/match between SMNDVI & MODIS
    smndvi_files = ret_smndvi_bin(directory_smndvi)
    modis_files = ret_modis_bin(directory_modis)

    # Load the slope & intercept to files
    with open('/data01/dlu12/LinRegModel/slope_grid.pkl', 'rb') as f:
        slope_grid = pickle.load(f)
    with open('/data01/dlu12/LinRegModel/intercept_grid.pkl', 'rb') as f:
        intercept_grid = pickle.load(f)

    # Call function to calculate evaluation metrics regarding Regression Model
    model_metrics(slope_grid, intercept_grid, smndvi_files, modis_files)
    
