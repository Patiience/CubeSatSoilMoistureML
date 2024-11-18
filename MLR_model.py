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
        new_rows = int((45 - (-45)) / 0.05)
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
def read_modis(directory_path):
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

        # Sort the files in the directory for matching purposes with other ancillary data
        files = os.listdir(year_path)
        sorted_files = sorted(files)

        # List to store file paths in order
        file_paths = []
    
        # Loop through the files in the year's directory
        for files in sorted_files:
            # File path
            file_path = os.path.join(year_path, files)

              # Print statement for tracking progress
            print(f"Loading data for {files}")

            # Append to list of file paths
            file_paths.append(file_path)

    # Return list of file paths for processing
    return file_paths


# Define function to compute Multiple Linear Regression Model
def mlr_model(modis_files, ancillary_grids, actual_SM, alpha):
  # Initialize the weights & intercept
  weight_modis = np.zeros((1800,7200))
  weight_elev = np.zeros((1800,7200))
  weight_landcov = np.zeros((1800,7200))
  weight_slopetype = np.zeros((1800,7200))
  weight_soiltext = np.zeros((1800,7200))
  intercept = np.zeros((1800,7200))

  # Load in the ancillary data grids
  elev_5km = ancillary_grids["Elevation"]
  landcover_5km = ancillary_grids["LandCover"]
  slopetype_5km = ancillary_grids["SlopeType"]
  soiltexture_5km = ancillary_grids["SoilTexture"]

  # Loop through daily files from MODIS dataset
  # Note: Using i to be able to also retrieve CYGNSS & SNR datasets at the same time
  # which requires the datasets to be the exact same length & dates
  for i in range(len(modis_files)):
    # Get the file path
    modis_path = modis_files[i]

    # Load in data from binary file, pickle & joblib throw errors, so use numpy
    # Will load it in as a memory map to avoid memory errors
    modis_grid = np.fromfile(modis_path, dtype='<f4')
    modis_grid = modis_grid.reshape((15000,36000))

    # Mask -9999 & exclude values outside of 0-1 range
    modis_grid = np.ma.masked_equal(modis_grid, -9999)
    modis_grid = np.ma.masked_outside(modis_grid, 0, 1)

    # Adjust domain of grid to match CYGNSS [-45, 45] for latitude
    lat1 = int((-45 + 60) / 0.01)
    lat2 = int((45 + 60)/ 0.01)
    lon1 = int((-180 + 180) / 0.01)
    lon2 = int((180 + 180) / 0.01)

    # Retrieve the correct domain matching CYGNSS
    modis_grid = modis_grid[lat1: lat2, lon1: lon2]

    # UPSCALE MODIS DATA TO 5KM
    # Define the size of each grid (5x5 for converting 1 km to 5 km)
    grid_size = 5

    # Calculate the dimensions of the new 5 km resolution array
    # Note: Using the ranges of the domain
    new_rows = int((45 - (-45)) / 0.05)
    new_cols = int((180 - (-180)) / 0.05)

    # Initialize an array to store the 5 km data
    modis_5km = np.zeros((new_rows, new_cols), dtype=np.float32)

    # Iterate through each 5x5 grid, calculate the mean, and store in the new array
    for i in range(new_rows):
        for j in range(new_cols):
            # Extract the 5x5 grid
            grid = modis_grid[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size]
            
            # Calculate the mean of the 5x5 grid
            grid_mean = np.mean(grid)
            
            # Assign the mean value to the corresponding position in the 5 km array
            modis_5km[i, j] = grid_mean

    # Calculate the predicted values for all the pixel locations
    y_pred = ((weight_modis * modis_5km) 
              + (weight_elev * elev_5km) 
              + (weight_landcov * landcover_5km) 
              + (weight_slopetype * slopetype_5km)
              + (weight_soiltext * soiltexture_5km)
              + intercept)

    # Calculate the error between Actual & Predicted
    error = actual_SM - y_pred

    # Update the weights & intercept accordingly based on the error
    weight_modis += alpha * error * modis_5km
    weight_elev += alpha * error * elev_5km
    weight_landcov += alpha * error * landcover_5km
    weight_slopetype += alpha * error * slopetype_5km
    weight_soiltext += alpha * error * soiltexture_5km
    intercept += alpha * error

  

    


   


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

  # Retrieve file paths to each data file from MODIS dataset
  modis_directory = "/data01/jyin/MODISNDVI/Gapfilling1kmNDVI"
  modis_files = read_modis(modis_directory)

  # Load in the actual data for Soil Moisture Retrievals
  actual_SM_directory = ""
  actual_SM = ""

  # Define hyperparameter learning rate
  alpha = 0.05

  # Call function to train SVR model
  weight, bias = mlr_model(modis_files, grids, actual_SM, alpha)
