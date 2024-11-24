# Import the necessary libraries
import matplotlib.pyplot as plt
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
              llcrnrlat=20, urcrnrlat=60, 
              llcrnrlon=-140, urcrnrlon=-60)

  # Draw coastlines and other map features
  m.drawcoastlines()
  m.drawcountries()
  m.drawparallels(np.arange(20, 61, 10), labels=[1,0,0,0])
  m.drawmeridians(np.arange(-140, -59, 20), labels=[0,0,0,1])

  # Create latitude and longitude arrays that match grid
  lon = np.linspace(-140, -60, ndvi_grid.shape[1])
  lat = np.linspace(20, 60, ndvi_grid.shape[0])

  # Convert the latitude and longitude grid to map projection coordinates
  lon, lat = np.meshgrid(lon, lat)
  x, y = m(lon, lat)

  # Mask values equal to -9999
  ndvi_grid = np.ma.masked_where(ndvi_grid == -9999, ndvi_grid)

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

  # Save the file to directory in data01
  plt.savefig(f'/data01/dlu12/NOAA20_TOC_Data/NDVI_TOC_Maps/NDVI_{date}.png')
  plt.close()

# Main function
if __name__ == "__main__":
  # Directory path for data
  file_path = "/data01/dlu12/NOAA20_TOC_Data/NDVI_TOC_Binaries"

  # Retrieve input for year to be mapped from the command line
  date = str(input("Enter date of file to be mapped (YYYYMMDD): "))
  file_name = f"NOAA20_TOC_NDVI_{date}.dat"
  file_path = os.path.join(file_path, file_name)

  # Print statement for tracking progress
  print(f"Creating map for {file_name}")

  # Load in data from binary file
  with open(file_path, 'rb') as f:
        ndvi_grid = pickle.load(f)
  
  # Call function to create map
  create_map(ndvi_grid, file_name)
