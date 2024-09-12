# Thank you, NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team for code reference
# Import libraries to be used
from netCDF4 import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#import cartopy.crs as ccrs

# Initialize file path to .nc file
directory_path = "/data01/jyin/NOAA20VI"

# Loop through 2019, 2021, 2022, 2023, 2024 directories
# for year in os.listdir(directory_path):
#   year_path = os.path.join(directory_path, year)
#   for date in os.listdir(year_path):
#     date_path = os.path.join(year_path, date)
#     for file in os.listdir(date_path):
#       file_path = os.path.join(date_path, file)
#       nc_file = Dataset(file_path)
file_path = "/data01/jyin/NOAA20VI/2019/20190715/VI-DLY-REG_v1r2_j01_s20190713_e20190713_c201907150108040.nc"

# Open the .nc file(s)
nc_file = Dataset(file_path)

# Retrieve NDVI_TOC data, in 2 dimensional (lat, lon)
nc_data = nc_file.variables['NDVI_TOC'][:,:]
lat = nc_file.variables['latitude'][:,:]
lon = nc_file.variables['longitude'][:,:]
# Unmask the data
nc_data = np.array(nc_data)

# Plot the data
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

# Convert the latitude and longitude grid to map projection coordinates
lon, lat = np.meshgrid(lon, lat)
x, y = m(lon, lat)

# Plot the data
m.pcolormesh(x, y, nc_data, shading='auto', cmap='viridis')  # Replace 'data' with your actual data

plt.title('NDVI for 7/15/2019')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.savefig('/home/dlu12/NDVI_test.png')
plt.show()
