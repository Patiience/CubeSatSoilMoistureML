# Import the necessary libraries
import matplotlib as plt
import pickle
from mpl_toolkits.basemap import Basemap
import os
import numpy as np

###################################
# BEGIN TO PLOT THE DATA ONTO MAP #
###################################
def create_map(ndvi_grid, file_name):
  # Plot the data using matplotlib
  plt.figure(figsize=(10, 5))

  # Create a Basemap instance
  m = Basemap(projection='cyl', resolution='c', 
              llcrnrlat=-90, urcrnrlat=90, 
              llcrnrlon=-180, urcrnrlon=180)

  # Draw coastlines and other map features
  m.drawcoastlines()
  m.drawcountries()
  m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
  m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])

  # Create latitude and longitude arrays that match grid
  lon = np.linspace(-180, 180, ndvi_grid.shape[1])
  lat = np.linspace(-90, 90, ndvi_grid.shape[0])

  # Convert the latitude and longitude grid to map projection coordinates
  lon, lat = np.meshgrid(lon, lat)
  x, y = m(lon, lat)

  # Plot the data
  mp = m.pcolormesh(x, y, ndvi_grid, shading='auto', cmap='viridis', vmin=0, vmax=1)  # Replace 'data' with your actual data

  # Add colorbar
  color_bar = m.colorbar(mp, location='right', pad="5%")
  color_bar.set_label("NDVI_TOC")

  # Get the YYYYMMDD of the map
  file_name_parts = file_name.split("_")
  date = (file_name_parts[-1].split("."))[0]

  # Update labels
  plt.title(f'NDVI for {date}')
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')

  # Save the file to directory in data01
  plt.savefig(f'/data01/dlu12/NDVI_Maps/NDVI_{date}.png')
  plt.close()

# Main function
if __name__ == "__main__":
  # Directory path for data
  directory_path = "/data01/dlu12/NDVI_Binaries"

  # Loop through files in directory
  for files in os.dir(directory_path):
    # File path
    file_path = os.path.join(directory_path, files)

    # Print statement for tracking progress
    print(f"Creating map for {file_path}")

    # Load in data from binary file
    with open(file_path, 'rb') as f:
          ndvi_grid = pickle.load(f)
    
    # Call function to create map
    create_map(ndvi_grid, files)
