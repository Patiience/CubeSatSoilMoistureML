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

        # Retrieve NDVI_TOC data, in 2 dimensional (lat, lon)
        nc_data = nc_file.variables['NDVI_TOC'][:,:]

        # Check and determine if lat and lon are 2-dimensional
        lat = nc_file.variables['latitude']
        lon = nc_file.variables['longitude']
        if len(lat.shape) == 1:
          lat = lat[:]
          lon = lon[:]
        elif len(lat.shape) == 2:
          lat = lat[:,:]
          lon = lon[:,:]

        # Unmask the data
        nc_data = np.array(nc_data)

        # Turn FILL_VALUE into -9999
        nc_data[nc_data == -32768] = -9999

        # Loop through nc_data and assign to grid accordingly
        for i in range(len(nc_data)):
          for j in range(len(nc_data[0])):
            # If already zero, don't need to update grid
            if nc_data[i][j] != 0:
              # Convert lat and lon to x,y grid points, and check for dimensions
              if len(lat.shape) == 1:
                # Normalize longitude
                lon_norm = (lon[j] + 180) % 360 - 180
                x = int(lat[i]/.01)
                y = int(lon_norm/.01)
              else:
                # Normalize longitude
                lon_norm = (lon[i][j] + 180) % 360 - 180
                x = int(lat[i][j]/.01)
                y = int(lon_norm/.01)

              # Assign to grid if it falls within the domain
              if (x >= 2000 and x < 6000) and (y >= -14000 and y < -6000):
                # Adjust x and y to fit into grid indices
                x = x - 2000
                y = y + 14000
                ndvi_grid[y][x] = nc_data[i][j]

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
