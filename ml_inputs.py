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
    if data_type != "LandCover":
      data = np.fromfile(file_path, dtype='>f4')  # '>f4' specifies big-endian float32
      data = data.reshape((15000,36000))
    else:
      data = np.fromfile(file_path, dtype=np.float32) # Default little-endian
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
