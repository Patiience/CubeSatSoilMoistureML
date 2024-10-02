# Thank you, NOAA/NESDIS/STAR Aerosols and Atmospheric Composition Science Team for code reference
# Import libraries to be used
from netCDF4 import Dataset
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function reads data and then dumps into a binary file
def calc_avg(directory_path):
    # Intialize array to store the averages
    md_avg = []
    tx_avg = []
    az_avg = []

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

            # Increment to avoid running this if statement again and also continue onto next iteration
            count += 1
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

        # NDVI grid is for 1 km resolution, which need to divide by .01
        # NWM domain is from 20, 60 lat & -140, -60 lon
        # Maryland: 37 to 39 lat, -79 to -75 lon
        # Texas: 25 to 36 lat, -106 to -93 lon
        # Arizona: 31 to 37 lat, -114 to -109 lon

        # Intialize lat and lon range for Maryland-ish region
        md_lat1 = (37 / 0.01).astype(np.int32)
        md_lat2 = (39 / 0.01).astype(np.int32)
        md_lon1 = (-79 / 0.01).astype(np.int32)
        md_lon2 = (-75 / 0.01).astype(np.int32)

        # Intialize lat and lon range for Texas-ish region
        tx_lat1 = (25 / 0.01).astype(np.int32)
        tx_lat2 = (36 / 0.01).astype(np.int32)
        tx_lon1 = (-106 / 0.01).astype(np.int32)
        tx_lon2 = (-93 / 0.01).astype(np.int32)

        # Intialize lat and lon range for Arizona-ish region
        az_lat1 = (31 / 0.01).astype(np.int32)
        az_lat2 = (37 / 0.01).astype(np.int32)
        az_lon1 = (-114 / 0.01).astype(np.int32)
        az_lon2 = (-109 / 0.01).astype(np.int32)

        # Adjust lat and lon values of each region to fit into grid indices
        md_lat1, md_lat2 = md_lat1 - 2000, md_lat2 - 2000
        md_lon1, md_lon2 = md_lon1 + 14000, md_lon2 + 14000
        tx_lat1, tx_lat2 = tx_lat1 - 2000, tx_lat2 - 2000
        tx_lon1, tx_lon2 = tx_lon1 + 14000, tx_lon2 + 14000
        az_lat1, az_lat2 = az_lat1 - 2000, az_lat2 - 2000
        az_lon1, az_lon2 = az_lon1 + 14000, az_lon2 + 14000

        # HERE DO THE INTERPOLATION METHOD !!!

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

            # Assign to grid using masked indices, which will get rid of corresponding indices
            maryland_grid = ndvi_grid_day[md_lat1: md_lat2 + 1, md_lon1: md_lon2 + 1]
            texas_grid = ndvi_grid_day[tx_lat1: tx_lat2 + 1, tx_lon1: tx_lon2 + 1]
            arizona_grid = ndvi_grid_day[az_lat1: az_lat2 + 1, az_lon1: az_lon2 + 1]

            # Calculate Averages for Each Region
            avg_md = np.mean(maryland_grid)
            avg_tx = np.mean(texas_grid)
            avg_az = np.mean(arizona_grid)

            # Append average value to array to later be plotted
            md_avg.append(avg_md)
            tx_avg.append(avg_tx)
            az_avg.append(avg_az)

        # Close the files
        nc_file1.close()
        nc_file2.close()

        # Update file names, now second file will be the start file
        file_name1 = file_name2

    # After looping through all the files, we have a list of all the averages of each day for each region
    # Initialize start and end dates, which were seen from files
    start_date = "2018-01-29"
    end_date = "2024-04-07"

    # Generate dates for the whole year (daily intervals)
    days = np.arange(np.datetime64(start_date), np.datetime64(end_date))

    # Plot the first dataset
    plt.plot(days, md_avg)
    plt.xlabel('Days')
    plt.ylabel('Average SMNDVI')
    plt.title('Average SMNDVI for Maryland Region from 2018-2024')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  # Format as 'Jan 01', 'Feb 15', etc.
    plt.gcf().autofmt_xdate()
    plt.savefig('/data01/dlu12/AVG_SMNDVI/Maryland.png')  # Save the first plot
    plt.clf()  # Clear the current plot

    # Plot the second dataset
    plt.plot(days, tx_avg)
    plt.xlabel('Days')
    plt.ylabel('Average SMNDVI')
    plt.title('Average SMNDVI for Texas Region from 2018-2024')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gcf().autofmt_xdate()
    plt.savefig('/data01/dlu12/AVG_SMNDVI/Texas.png')  # Save the second plot
    plt.clf()  # Clear the current plot

    # Plot the third dataset
    plt.plot(days, az_avg)
    plt.xlabel('Days')
    plt.ylabel('Average SMNDVI')
    plt.title('Average SMNDVI for Arizona Region from 2018-2024')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gcf().autofmt_xdate()
    plt.savefig('/data01/dlu12/AVG_SMNDVI/Arizona.png')  # Save the third plot
    plt.clf()  # Clear the current plot

# Main function
if __name__ == "__main__":
    # Directory path for data
    directory_path = "/data/jyin/SMNDVI"

    # Read the data
    calc_avg(directory_path)
