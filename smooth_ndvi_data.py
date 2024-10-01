# Thank you, NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team for code reference
# Import libraries to be used
from netCDF4 import Dataset
import numpy as np
import os
import pickle
from datetime import datetime, timedelta

# Function reads data and then dumps into a binary file
def read_map_data(directory_path):
  # Variable to just track when interpolation should begin
  count = 0

  # Sort directory to ensure we get correct order of files
  directory = os.listdir(directory_path)
  sorted_directory = sorted(directory)


  # Variables to keep track of files being read
  file_name1 = ""
  file_name2 = ""

  # Loop through all the files in the directory
  for files in sorted_directory:
    # If looping through just the first file
    if count == 0:
      # Update first file_name
      file_name1 = files

      # For testing purposes and to ensure no files were skipped
      print(f"File Start Check: {file_name1}")

      # Split file name into parts to parse
      file_parts1 = file_name1.split(".")

      # Get the year of the file
      file_year = (file_parts1[-3])[1:5]

      # Check if within 2019-2023 years to begin collection, else don't and continue until find first 2019 file
      if file_year == "2019":
        # Increment to avoid running this if statement again and also continue onto next iteration
        count += 1
        continue
      else:
        continue

    # Set filename for second file
    file_name2 = files

    # Get file paths for both files
    file_path1 = os.path.join(directory_path, file_name1)
    file_path2 = os.path.join(directory_path, file_name2)

    # Print statement to track progress when running program
    print(f"Reading data from {file_name1} and {file_name2}")

    # Create Dataset Object & open the .nc file(s)
    nc_file1 = Dataset(file_path1)
    nc_file2 = Dataset(file_path2)

    # Get the start date of only the first file and removing the extra characters behind
    file_date = str(getattr(nc_file1, 'time_coverage_start', None))[:8]

    # Initialize NDVI grid, for 1 km resolution, which divide by .01
    # NWM domain is from 20, 60 lat & -140, -60 lon
    # Note: grid is latitude by longitude
    ndvi_grid1 = np.zeros((4000, 8000))
    ndvi_grid2 = np.zeros((4000, 8000))

    # Retrieve NDVI_TOC data, in 2-dimensional (lat, lon)
    nc_data1 = nc_file1.variables['SMN'][:]
    nc_data2 = nc_file2.variables['SMN'][:]

    # Unmask the data
    nc_data1 = np.array(nc_data1)
    nc_data2 = np.array(nc_data2)

    # Retrive latitude and longitude data, which could be 2D or 1D, but code will get full array regardless
    lat1 = nc_file1.variables['latitude'][:]
    lon1 = nc_file1.variables['longitude'][:]
    lat2 = nc_file2.variables['latitude'][:]
    lon2 = nc_file2.variables['longitude'][:]

    # If array is 1D, make it so it is 2D in order to carry out functions later
    if len(lat1.shape) == 1 and len(lat2.shape) == 1:
      # Broadcast arrays to 2d array
      lat1 = np.tile(lat1[:,np.newaxis], (1, nc_data1.shape[1]))
      lon1 = np.broadcast_to(lon1[np.newaxis, :], (nc_data1.shape[0], nc_data1.shape[1]))
      lat2 = np.tile(lat2[:,np.newaxis], (1, nc_data2.shape[1]))
      lon2 = np.broadcast_to(lon2[np.newaxis, :], (nc_data2.shape[0], nc_data2.shape[1]))

    # Copy the arrays, as tile() and broadcast_to() functions lead to read-only for some reason
    lat1 = np.copy(lat1)
    lon1 = np.copy(lon1)
    lat2 = np.copy(lat2)
    lon2 = np.copy(lon2)

    # Convert lat and lon to grid indices
    x_indices1 = (lat1 / 0.01).astype(np.int32)
    y_indices1 = (lon1 / 0.01).astype(np.int32)
    x_indices2 = (lat2 / 0.01).astype(np.int32)
    y_indices2 = (lon2 / 0.01).astype(np.int32)

    # Mask to ignore indicies that do not fall within NWM domain
    mask1 = (x_indices1 >= 2000) & (x_indices1 < 6000) & (y_indices1 >= -14000) & (y_indices1 < -6000)
    mask2 = (x_indices2 >= 2000) & (x_indices2 < 6000) & (y_indices2 >= -14000) & (y_indices2 < -6000)

    # Adjust x and y indices to fit into grid indices
    adjusted_x_indices1 = x_indices1 - 2000
    adjusted_y_indices1 = y_indices1 + 14000
    adjusted_x_indices2 = x_indices2 - 2000
    adjusted_y_indices2 = y_indices2 + 14000

    # Assign to grid using masked indices, which will get rid of corresponding indices
    ndvi_grid1[adjusted_x_indices1[mask1], adjusted_y_indices1[mask1]] = nc_data1[mask1]
    ndvi_grid2[adjusted_x_indices2[mask2], adjusted_y_indices2[mask2]] = nc_data2[mask2]

    # Delete numpy arrays to free memory
    del adjusted_x_indices1, adjusted_y_indices1, adjusted_x_indices2, adjusted_y_indices2, lat1, lon1, lat2, lon2

    # HERE DO THE INTERPOLATION METHOD !!!
    # Initialize Datetime object to add correctly to the date later
    date_format = "%Y%m%d"
    start_date_obj = datetime.strptime(file_date, date_format)

    # Initialize values for interpolation
    val_week1 = 6
    val_week2 = 0
    ndvi_grid_day = []
    # Loop 7 times for 7 days in between weeks
    for i in range(7):
      # Perform the weighted interpolation to get grid for the day
      ndvi_grid_day = ((val_week1 * ndvi_grid1) + (val_week2 * ndvi_grid2))/6
      # Prepare values for next interpolation
      val_week1 -= 1
      val_week2 += 1

      # Increment the date accordingly and convert to string for binary file name
      date_obj = start_date_obj + timedelta(days=i)
      date = date_obj.strftime(date_format)

      # For each file, dump into binary file in data01 directory
      binary_path = '/data01/dlu12/NDVI_Binaries'
      binary_file = f'SMNDVI_{date}.dat'
      binary_path = os.path.join(binary_path, binary_file)

      with open(binary_path, 'wb') as f:
        pickle.dump(ndvi_grid_day, f)

    # Close the files
    nc_file1.close()
    nc_file2.close()

    # Split second file name into parts to parse
    file_parts2 = file_name2.split(".")
    # Get the year of the second file
    file_year2 = (file_parts2[-3])[1:5]

    # Check to see if second file is beginning of 2024, if so we can end collection, else continue
    if file_year2 == "2024":
      break

    # Update file names, now second file will be the start file
    file_name1 = file_name2


# Main function
if __name__ == "__main__":
  # Initialize file path to .nc file
  directory_path = "/data/jyin/SMNDVI"

  # Read the data
  read_map_data(directory_path)
