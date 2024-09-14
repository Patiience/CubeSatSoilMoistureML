# Thank you, NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team for code reference
# Import libraries to be used
from netCDF4 import Dataset
import numpy as np
import os
import pickle

# Function reads data and then dumps into a binary file
def read_map_data(directory_path):
  # Loop through 2019, 2021, 2022, 2023, 2024 directories
  for year in os.listdir(directory_path):
    # Path to year directory
    year_path = os.path.join(directory_path, year)
    # Loop through each date directory
    for date in os.listdir(year_path):
      # Path to date directory
      date_path = os.path.join(year_path, date)
      # Loop through files in date directory
      for files in os.listdir(date_path):
        # Path to file
        file_path = os.path.join(date_path, files)

        #file_path = "/data01/jyin/NOAA20VI/2019/20190715/VI-DLY-REG_v1r2_j01_s20190713_e20190713_c201907150108040.nc"

        # Create Dataset Object & open the .nc file(s)
        nc_file = Dataset(file_path)

        # Initialize NDVI grid, for 1 km resolution
        ndvi_grid = np.zeros((18000, 36000))

        # Retrieve NDVI_TOC data, in 2 dimensional (lat, lon), as well as lat and lon coordinates
        nc_data = nc_file.variables['NDVI_TOC'][:,:]
        lat = nc_file.variables['latitude'][:,:]
        lon = nc_file.variables['longitude'][:,:]

        # Unmask the data
        nc_data = np.array(nc_data)

        # Turn FILL_VALUE into -9999
        nc_data[nc_data == -32768] = -9999

        # Loop through nc_data and assign to grid accordingly
        for i in range(len(nc_data)):
          for j in range(len(nc_data[0])):
            # If already zero, don't need to update grid
            if nc_data[i][j] != 0:
              # Convert lat and lon to x,y grid points
              x = lat[i][j]/.01
              y = lon[i][j]/.01

              # Assign to grid
              ndvi_grid[y][x] = nc_data[i][j]

        # For each file, dump into binary file in data01 directory
        binary_path = '/data01/dlu12/NDVI_Binaries'
        binary_file = f'NOAA20_TOC_NDVI_{files}.dat'
        binary_path = os.path.join(data_path, binary_file)

        with open(binary_path, 'wb') as f:
          pickle.dump(ndvi_grid, f)

        # Close the file
        nc_file.close()

# Main function
if __name__ == "__main__":
  # Initialize file path to .nc file
  directory_path = "/data01/jyin/NOAA20VI"

  # Read the data
  read_map_data(directory_path)
