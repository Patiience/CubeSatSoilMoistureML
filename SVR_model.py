# Import libraries to be used
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


# Function reads data and then dumps into a binary file
def read_data(directory_path, files):
  # Intialize dictionary to store grids
  grids = {}

  # Loop through each of the Ancillary Data
  for data_type, filename in files.items():
    # Merge filename to create file path
    file_path = os.path.join(directory_path, filename)

    # Read file and reshape afterward
    if data_type == "Elevation" or data_type == "SlopeType":
      data = np.fromfile(file_path, dtype='>f4')  # '>f4' specifies big-endian float32
      data = data.reshape((15000,36000))
    elif data_type == "SoilTexture":
      data = np.fromfile(file_path, dtype='>i4')  # '>i4' specifies big-endian int32
      data = data.reshape((15000,36000))
    elif data_type == "LandCover":
      data = np.fromfile(file_path, dtype=np.int32) # Default little-endian
      data = data.reshape((18000,36000)) # Landcover covers whole global domain
      data = np.flipud(data)  # Flip LandCover data since it's upside down

      # Intialize lat and lon range to match domain for others
      # -60 to 90 N for lat & -180 to 180 E lon
      # Addition is to adjust lat and lon values to adjust in correspondance to MODIS dataset indices
      # -90 to 90 N for lat & -180 to 180 E lon
      lat1 = int((-60 + 90) / 0.01)
      lat2 = int((90 + 90)/ 0.01)
      lon1 = int((-180 + 180) / 0.01)
      lon2 = int((180 + 180) / 0.01)

      # Retrieve the correct domain
      data = data[lat1: lat2, lon1: lon2]
    
    # Mask -9999
    data = np.ma.masked_equal(data, -9999)

    # Adjust domain of grid to match CYGNSS [-45, 45] for latitude
    lat1 = int((-45 + 60) / 0.01)
    lat2 = int((45 + 60)/ 0.01)
    lon1 = int((-180 + 180) / 0.01)
    lon2 = int((180 + 180) / 0.01)

    # Retrieve the correct domain matching CYGNSS
    data = data[lat1: lat2, lon1: lon2]

    # Need to upscale the data from 1km to 5km
    if data_type == "Elevation" or data_type == "SlopeType":
      # Define the size of each grid (5x5 for converting 1 km to 5 km)
      grid_size = 5

      # Calculate the dimensions of the new 5 km resolution array
      # Note: Using the ranges of the domain
      new_rows = int((45 - (-45)) / 0.05)
      new_cols = int((180 - (-180)) / 0.05)

      # Initialize an array to store the 5 km data
      data_5km = np.zeros((new_rows, new_cols), dtype=np.float32)

      # Iterate through each 5x5 grid, calculate the mean, and store in the new array
      for i in range(new_rows):
          for j in range(new_cols):
              # Extract the 5x5 grid
              grid = data[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
              
              # Calculate the mean of the 5x5 grid
              grid_mean = np.mean(grid)
              
              # Assign the mean value to the corresponding position in the 5 km array
              data_5km[i, j] = grid_mean

    elif data_type == "SoilTexture" or data_type == "LandCover":
        # Define the size of each grid (5x5 for converting 1 km to 5 km)
        grid_size = 5

        # Calculate the dimensions of the new 5 km resolution array
        # Note: Using the ranges of the domain
        new_rows = int((90 - (-60)) / 0.05)
        new_cols = int((180 - (-180)) / 0.05)

        # Initialize an array to store the 5 km data
        data_5km = np.zeros((new_rows, new_cols), dtype=np.float32)

        # Iterate through each 5x5 grid, calculate the mean, and store in the new array
        for i in range(new_rows):
            for j in range(new_cols):
                # Extract the 5x5 grid
                grid = data[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
                
                # Calculate the most frequent value of the 5x5 grid
                values, counts = np.unique(grid.flatten(), return_counts=True)
                mode_value = values[np.argmax(counts)]
                
                # Assign the mean value to the corresponding position in the 5 km array
                data_5km[i, j] = mode_value

    # Append to grid dictionary
    grids[data_type] = data_5km

  # Return dictionary
  return grids


# Define function to process MODIS data
def process_modis(directory_path):
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

            # Mask -9999 & exclude values outside of 0-1 range
            ndvi_grid = np.ma.masked_equal(ndvi_grid, -9999)
            ndvi_grid = np.ma.masked_outside(ndvi_grid, 0, 1)

            # UPSCALE MODIS DATA TO 5KM
            # Define the size of each grid (5x5 for converting 1 km to 5 km)
            grid_size = 5

            # Calculate the dimensions of the new 5 km resolution array
            # Note: Using the ranges of the domain
            new_rows = int((90 - (-60)) / 0.05)
            new_cols = int((180 - (-180)) / 0.05)

            # Initialize an array to store the 5 km data
            data_5km = np.zeros((new_rows, new_cols), dtype=np.float32)

            # Iterate through each 5x5 grid, calculate the mean, and store in the new array
            for i in range(new_rows):
                for j in range(new_cols):
                    # Extract the 5x5 grid
                    grid = data[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
                    
                    # Calculate the mean of the 5x5 grid
                    grid_mean = np.mean(grid)
                    
                    # Assign the mean value to the corresponding position in the 5 km array
                    data_5km[i, j] = grid_mean

            ## WHAT TO DO WITH THE UPSCALED DATA ##


# Define function to compute Support Vector Regression Model
def svr_model(training_data, y, X, C, epsilon, kernel_c, kernel_d):
    # Polynomial kernel computation (K(x_i, x_j) = (x_i^T x_j + c)^d)
    K = (np.dot(X, X.T) + kernel_c) ** kernel_d
    
    # Set up P, q, G, h, A, b for Quadratic Programming
    n_samples = X.shape[0]
    P = np.vstack([
        np.hstack([K, -K]),
        np.hstack([-K, K])
    ])

    # Set up the target as per SVR's epsilon-insensitive loss
    q = epsilon * np.ones(2 * n_samples) - np.hstack([y, -y])

    # Constraints: -alpha <= 0 and alpha <= C
    G = np.vstack([-np.eye(2 * n_samples), np.eye(2 * n_samples)])
    h = np.hstack([np.zeros(2 * n_samples), C * np.ones(2 * n_samples)])

    # A and b for equality constraint (sum(alpha - alpha_star) = 0)
    A = np.hstack([np.ones(n_samples), -np.ones(n_samples)]).reshape(1, -1)
    b = np.zeros(1)

    # Solve the quadratic programming problem
    qp_solution = solve_qp(P, q, G, h, A, b, solver="osqp")
    
    # Split the solution into alpha and alpha_star
    alpha = qp_solution[:n_samples]
    alpha_star = qp_solution[n_samples:]
    
    # Retrieve indices for support vectors (where alpha or alpha_star > 0)
    sv_indices = np.where((alpha - alpha_star) > 1e-5)[0]
    sv_alpha = alpha[sv_indices]
    sv_alpha_star = alpha_star[sv_indices]
    sv_X = X[sv_indices]
    sv_y = y[sv_indices]
    
    # Compute the weights vector if using a linear kernel
    w = np.dot((sv_alpha - sv_alpha_star), sv_X)
    
    # Compute the bias term (intercept)
    bias = np.mean(sv_y - np.dot(sv_X, w))

    return w, bias


# Main function
if __name__ == "__main__":
  # Initialize file path to .nc file
  directory_path = "/data01/jyin/ForInterns"

  # Intialize dictionaries to correspond to filenames
  files = {"Elevation": "elev_GTOPO30.1gd4r",
           "LandCover": "landcover_IGBP_NCEP.1gd4r",
           "SlopeType": "slopetype_NCEP.1gd4r",
           "SoilTexture": "soiltexture_STATSGO-FAO.1gd4r"}

  # Read the data
  grids = read_ancillary_data(directory_path, files)

  # Process Modis
  modis_directory = "/data01/jyin/MODISNDVI/Gapfilling1kmNDVI"
  process_modis(modis_directory)

  # Call function to train SVR model
  weight, bias = svr_model()
