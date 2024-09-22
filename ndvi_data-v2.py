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

        # Split file name into parts to parse
        file_parts = files.split("_")

        # Get the start and end dates of the file
        file_start = (file_parts[3])[1:]
        file_end = (file_parts[4])[1:]

        # Check if it matches the date of the directory, and if so retrieve its data
        # If it doesn't match, move onto the next file
        if (file_start == date) and (file_end == date):
          # Print statement to track progress when running program
          print(f"Reading data from {file_path}")

          # Create Dataset Object & open the .nc file(s)
          nc_file = Dataset(file_path)

          # Initialize NDVI grid, for 1 km resolution, which divide by .01
          # CONUS domain is from 20, 60 lat & -140, -60 lon
          # Note: grid is latitude by longitude
          ndvi_grid = np.zeros((4000, 8000))

          # Retrieve NDVI_TOC data, in 2-dimensional (lat, lon)
          nc_data = nc_file.variables['NDVI_TOC'][:]

          # Unmask the data
          nc_data = np.array(nc_data)

          # Turn FILL_VALUE into -9999
          nc_data[nc_data == -32768] = -9999

          # Retrive latitude and longitude data, which could be 2D or 1D, but code will get full array regardless
          lat = nc_file.variables['latitude'][:]
          lon = nc_file.variables['longitude'][:]

          # If array is 1D, make it so it is 2D in order to carry out functions later
          if len(lat.shape) == 1:
            # Broadcast arrays to 2d array
            lat = np.tile(lat[:,np.newaxis], (1, nc_data.shape[1]))
            lon = np.broadcast_to(lon[np.newaxis, :], (nc_data.shape[0], nc_data.shape[1]))

          # Copy the arrays, as tile() and broadcast_to() functions lead to read-only for some reason
          lat = np.copy(lat)
          lon = np.copy(lon)

          # Update values less than -180 and offset by 360
          lon[lon < -180] += 360

          # Convert lat and lon to grid indices
          x_indices = (lat / 0.01).astype(int)
          y_indices = (lon / 0.01).astype(int)

          # Mask to ignore indicies that do not fall within NWM domain
          mask = (x_indices >= 2000) & (x_indices < 6000) & (y_indices >= -14000) & (y_indices < -6000)

          # Adjust x and y indices to fit into grid indices
          adjusted_x_indices = x_indices - 2000
          adjusted_y_indices = y_indices + 14000

          # Assign to grid using masked indices, which will get rid of corresponding indices
          ndvi_grid[adjusted_x_indices[mask], adjusted_y_indices[mask]] = nc_data[mask]

          # For each file, dump into binary file in data01 directory
          binary_path = '/data01/dlu12/NDVI_Binaries'
          binary_file = f'NOAA20_TOC_NDVI_{date}.dat'
          binary_path = os.path.join(binary_path, binary_file)

          with open(binary_path, 'wb') as f:
            pickle.dump(ndvi_grid, f)

          # Close the file
          nc_file.close()

          # Break out of loop, since there is only one file to read for each date directory
          break


# Main function
if __name__ == "__main__":
  # Initialize file path to .nc file
  directory_path = "/data01/jyin/NOAA20VI"

  # Read the data
  read_map_data(directory_path)
