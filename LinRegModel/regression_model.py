# Import other necessary libraries
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

# Define function to retrieve the data from the SMNDVI binaries
def ret_smndvi_bin(directory_path):
    # Initialize matrices to be used to calculate Linear Regression Model later
    sum_smndvi_x = np.zeros((4000,8000))
    sum_smndvi_x2 = np.zeros((4000,8000))
    N = np.zeros((4000,8000))

    # List used for testing later
    file_list = []

    # Sort directory to ensure we get correct order of files
    directory = os.listdir(directory_path)
    sorted_directory = sorted(directory)

    # Loop through files in directory
    for files in sorted_directory:
        # File path
        file_path = os.path.join(directory_path, files)

         # Print statement for tracking progress
        print(f"Loading data for {files}")

        # Load in data from binary file
        with open(file_path, 'rb') as f:
            ndvi_grid = pickle.load(f)

        # Convert to numpy array & mask -999, excluding values outside of 0-1 range
        ndvi_grid = np.array(ndvi_grid, dtype=np.float32)
        ndvi_grid = np.ma.masked_equal(ndvi_grid, -999)
        ndvi_grid = np.ma.masked_outside(ndvi_grid, 0, 1)

        # Update sum(X) & sum(X^2) & N, ignoring masked values
        sum_smndvi_x = sum_smndvi_x + ndvi_grid.filled(0)  # Use 0 for masked values in sum
        sum_smndvi_x2 = sum_smndvi_x2 + (ndvi_grid ** 2).filled(0)
        N = N + 1

        # Append file to files list for evaluation later
        file_list.append(file_path)

    # Return Matrices sum(X) & sum(X^2)
    return sum_smndvi_x, sum_smndvi_x2, N, file_list


# Define function to retrieve the data from the MODIS binaries
def ret_modis_bin(directory_path, N1):
    # Initialize matrices to be used to calculate Linear Regression Model later
    sum_modis_y = np.zeros((4000,8000))
    N = np.zeros((4000,8000))

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

              # Print statement for tracking progress
            print(f"Loading data for {files}")

            # Load in data from binary file, pickle & joblib throw errors, so use numpy
            # Will load it in as a memory map to avoid memory errors
            ndvi_grid = np.fromfile(file_path, dtype='<f4')
            ndvi_grid = ndvi_grid.reshape((15000,36000))

            # Intialize lat and lon range for the NWM domain
            # NWM domain is from 20, 60 lat & -140, -60 lon
            # Addition is to adjust lat and lon values to match MODIS dataset indices
            # -60 to 90 N for lat & -180 to 180 E lon
            lat1 = int((20 + 60) / 0.01)
            lat2 = int((60 + 60)/ 0.01)
            lon1 = int((-140 + 180) / 0.01)
            lon2 = int((-60 + 180) / 0.01)

            # Retrieve the NWM domain through indexing
            ndvi_grid = ndvi_grid[lat1: lat2, lon1: lon2]

            # Mask -9999 & exclude values outside of 0-1 range
            ndvi_grid = np.ma.masked_equal(ndvi_grid, -9999)
            ndvi_grid = np.ma.masked_outside(ndvi_grid, 0, 1)

             # Update sum(Y) & N, ignore mask values by setting equal to zero
            sum_modis_y = sum_modis_y + ndvi_grid.filled(0)
            N = N + 1

            # Append file path to files list for evaluation later
            file_list.append(file_path)
    
    # Check to see if two N matrices from both datasets are equal
    equal_N = np.array_equal(N1, N)
    # If not, throw an error
    if equal_N != True:
        raise Exception("Values of N are not equal")

    # Return sum(Y) & N & list of files
    return sum_modis_y, N, file_list

# Define function to calculate sum(XY)
def calc_xy(smndvi_files, modis_files):
    # Initialize matrices to calculate sum(XY)
    sum_xy = np.zeros((4000,8000))

    # Calculate Predicted MODIS value for each SMNDVI file, then compare to actual value of corresponding MODIS
    for i in range(len(smndvi_files)):
        # Load in file paths for SMNDVI and MODIS data
        file_smndvi = smndvi_files[i]
        file_modis = modis_files[i]

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

        # Update sum(XY)
        sum_xy = sum_xy + (smndvi_grid * modis_grid).filled(0)

    # Return sum(XY)
    return sum_xy
        

# Define function to perform evaluation for the Linear Regression Model
def model_metrics(slope_grid, intercept_grid, smndvi_files, modis_files):
    # Intialize list to store MSE
    mse_values = []

    # Calculate Predicted MODIS value for each SMNDVI file, then compare to actual value of corresponding MODIS
    for i in range(len(smndvi_files)):
        # Load in file paths for SMNDVI and MODIS data
        file_smndvi = smndvi_files[i]
        file_modis = modis_files[i]

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

        # Calculate the Mean Squared Error
        mse = np.mean((modis_grid - modis_pred) ** 2)
        mse_values.append(mse)

    # Calculate overall mean MSE
    mean_mse = np.mean(mse_values)

    # Plot the MSE values for better visualization
    plt.figure(figsize=(10, 6))
    plt.plot(mse_values, marker='o', linestyle='-', color='blue', label='MSE')
    plt.title('MSE for Predictions for Daily Files')
    plt.xlabel('Files (in Chronological Order)')  # Label for the automatically generated x-axis
    plt.ylabel('MSE')
    plt.text(0.5, 1.05, f'Mean MSE: {mean_mse}', fontsize=10, ha='center', transform=plt.gca().transAxes) # Add some text regarding Mean MSE
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('/data01/dlu12/LinRegModel/MSE.png')


# Main function to train and test linear regression model
if __name__ == "__main__":
    # Directory path for SMNDVI binaries retrieved earlier & MODIS NDVI
    directory_smndvi = "/data01/dlu12/All_SMNDVI_Binaries"
    directory_modis = "/data01/jyin/MODISNDVI/Gapfilling1kmNDVI"

    # Retrieve matrices summing up values from each dataset to calculate slope and intercept
    # Lists should be in order from 2018-2024 and indices should correspond/match between SMNDVI & MODIS
    sum_smndvi_x, sum_smndvi_x2, N, smndvi_files = ret_smndvi_bin(directory_smndvi)
    sum_modis_y, N, modis_files = ret_modis_bin(directory_modis, N)

    # Calculate sum(XY) & sum(X)^2
    sum_xy = calc_xy(smndvi_files, modis_files)
    sum_squared_x = sum_smndvi_x ** 2

    # Now that we have all the necessary matrices, calculate slope & intercept
    # Note: Slope & intercept are also matrices
    intercept_grid = ((sum_modis_y * sum_smndvi_x2) - (sum_smndvi_x * sum_xy)) / ((N * sum_smndvi_x2) - sum_squared_x)
    slope_grid = ((N * sum_xy) - (sum_smndvi_x * sum_modis_y)) / ((N * sum_smndvi_x2) - sum_squared_x)

    # Save the slope & intercept to files
    with open('/data01/dlu12/LinRegModel/slope_grid.pkl', 'wb') as f:
        pickle.dump(slope_grid, f)
    with open('/data01/dlu12/LinRegModel/intercept_grid.pkl', 'wb') as f:
        pickle.dump(intercept_grid, f)

    # Call function to calculate evaluation metrics regarding Regression Model
    model_metrics(slope_grid, intercept_grid, smndvi_files, modis_files)
    
