# Import libraries to be used
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Function reads data and then dumps into a binary file
def read_data(directory_path, files):
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

    # Need to upscale the data from 1km to 5km
    if data_type == "Elevation" or data_type == "SlopeType":
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


# Define function to plot data if needed
def plot(data, data_type):
    # Calculate & print some summary statistics
    mean = np.mean(data) 
    median = np.median(data) 
    minimum = np.min(data) 
    maximum = np.max(data) 

    print(f"{data_type} Mean: {mean},")
    print(f"{data_type} Median: {median},")
    print(f"{data_type} Min: {minimum},")
    print(f"{data_type} Max: {maximum}")

    # Plot data to visualize it
    # Plot the data using matplotlib
    plt.figure(figsize=(10, 5))

    # Create a Basemap instance
    m = Basemap(projection='cyl', resolution='c', 
                llcrnrlat=-60, urcrnrlat=90, 
                llcrnrlon=-180, urcrnrlon=180)

    # Draw coastlines and other map features
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-60, 91, 10), labels=[1,0,0,0])
    m.drawmeridians(np.arange(-180, 181, 20), labels=[0,0,0,1])

    # Create latitude and longitude arrays that match grid
    lon = np.linspace(-180, 180, data.shape[1])
    lat = np.linspace(-60, 90, data.shape[0])

    # Convert the latitude and longitude grid to map projection coordinates
    lon, lat = np.meshgrid(lon, lat)
    x, y = m(lon, lat)

    # Plot the data
    mp = m.pcolormesh(x, y, data, shading='auto', cmap='viridis')  # Replace 'data' with your actual data

    # Add colorbar
    color_bar = m.colorbar(mp, location='right', pad="5%")
    color_bar.set_label(f"{data_type}")

    # Update labels
    plt.title(f'{data_type} Data')

    # Save the file to directory in data01
    plt.savefig(f'/home/dlu12/{data_type}.png')
    plt.close()


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
  read_data(directory_path, files)

  # Process Modis
  modis_directory = "/data01/jyin/MODISNDVI/Gapfilling1kmNDVI"
  process_modis(modis_directory)
