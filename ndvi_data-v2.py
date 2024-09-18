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

        # Print statement to track progress when running program
        print(f"Reading data from {file_path}")

        # Create Dataset Object & open the .nc file(s)
        nc_file = Dataset(file_path)

        # Initialize NDVI grid, for 1 km resolution, which divide by .01
        # CONUS domain is from 20, 60 lat & -140, -60 lon
        # Note: grid is longitude by latitude
        ndvi_grid = np.zeros((8000, 4000))

        # Retrieve NDVI_TOC data, in 2-dimensional (lat, lon)
        nc_data = nc_file.variables['NDVI_TOC'][:,:]

        # Check and determine if lat and lon are 2-dimensional
        lat = nc_file.variables['latitude']
        lon = nc_file.variables['longitude']
        if len(lat.shape) == 1:
          # Broadcast arrays to 2d array
          lat = np.tile(lat[:,np.newaxis], (1, nc_data.shape[1]))
          lon = np.broadcast_to(lat[np.newaxis, :], (nc_data.shape[0], nc_data.shape[1]))
        elif len(lat.shape) == 2:
          lat = lat[:,:]
          lon = lon[:,:]

        # Unmask the data
        nc_data = np.array(nc_data)

        # Turn FILL_VALUE into -9999
        nc_data[nc_data == -32768] = -9999

        # Update values less than -180 and offset by 360
        lon[lon < -180] += 360

        # Convert lat and lon to grid indices
        x_indices = (lat / 0.01).astype(int)
        y_indices = (lon / 0.01).astype(int)

        # Mask to ignore zero values and indicies that do not fall within NWM domain
        mask = (nc_data != 0) & (x_indices >= 2000) & (x_indices < 6000) & (y_indices >= -14000) & (y_indices < -6000)

        # Adjust x and y indices to fit into grid indices
        adjusted_x_indices = x_indices - 2000
        adjusted_y_indices = y_indices + 14000

        # Assign to grid using masked indices, which will get rid of corresponding indices
        ndvi_grid[adjusted_y_indices[mask], adjusted_x_indices[mask]] = nc_data[mask]

        # For each file, dump into binary file in data01 directory
        binary_path = '/data01/dlu12/NDVI_Binaries'
        binary_file = f'NOAA20_TOC_NDVI_{files}.dat'
        binary_path = os.path.join(data_path, binary_file)

        with open(binary_path, 'wb') as f:
          pickle.dump(ndvi_grid, f)

        # Close the file
        nc_file.close()

        break
      break
    break


# Main function
if __name__ == "__main__":
  # Initialize file path to .nc file
  directory_path = "/data01/jyin/NOAA20VI"

  # Read the data
  read_map_data(directory_path)