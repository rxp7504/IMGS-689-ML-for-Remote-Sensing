# IMGS-689: Machine Learning for Remote Sensing
# Homework #1 - Exploratory Data Analysis (EDA)
# Author: Robert-Jason Pearsall

import numpy as np
import cmocean as cmo
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from numpy.ma.core import zeros_like
from scipy.ndimage import rotate
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import gaussian_kde
from itertools import combinations


# ----------------Plot band function----------------------
def plot_band():

    # Number of rows and columns for the plot
    rows, cols = 2, 6

    # initialize the subplot structure
    fig, axes = plt.subplots(rows, cols,figsize=(10,5))

    # flatten axes for iteration
    axes = axes.flatten()

    # Loop through each channel and plot it
    for i in range(data_new.shape[2]):

        channel = data_new[:,:,i]
        ax = axes[i]  # Select the current axis
        im = ax.imshow(channel, cmap='viridis', aspect='auto' )
        ax.axis('off')
        ax.set_title(f"{wl[i]}nm ",fontsize=9)  # Set the title for each subplot

    # Add a vertical colorbar for the entire montage
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.05)

    # Set a title for the entire figure
    fig.suptitle("Sentinel-2 Bands", fontsize=16)

    plt.show()

# ----------- Calculate Band Statistics Function----------
def calculate_band_statistics(data):

    if len(data.shape) > 2:
        data = data.reshape(-1, data.shape[2])



    data_mean = np.mean(data,axis=0)
    data_std = np.std(data,axis=0)
    data_min = np.min(data,axis=0)
    data_max = np.max(data,axis=0)
    data_q1 = np.percentile(data,25,axis=0)
    data_median = np.median(data,axis=0)
    data_q3 = np.percentile(data,75,axis=0)
    data_skew = skew(data,axis=0)
    data_kurt = kurtosis(data,axis=0)

    # data_mean = np.zeros(data.shape[1])
    # data_std = np.zeros(data.shape[1])
    # data_min = np.zeros(data.shape[1])
    # data_max = np.zeros(data.shape[1])
    # data_q1 = np.zeros(data.shape[1])
    # data_median = np.zeros(data.shape[1])
    # data_q3 = np.zeros(data.shape[1])
    # data_skew = np.zeros(data.shape[1])
    # data_kurt = np.zeros(data.shape[1])

    # for i in range(data.shape[1]):
    #     band = data[:,i]
    #     band = band[band>0]
    #
    #     data_mean[i] = np.mean(band)
    #     data_std[i] = np.std(band)
    #     data_min[i] = np.min(band)
    #     data_max[i] = np.max(band)
    #     data_q1[i] = np.percentile(band, 25)
    #     data_median[i] = np.median(band)
    #     data_q3[i] = np.percentile(band, 75)
    #     data_skew[i] = skew(band)
    #     data_kurt[i] = kurtosis(band)

    stats = np.vstack([data_mean,data_std,data_min,data_max,data_q1,data_median,data_q3,data_skew,data_kurt]).T

    # Format the matrix to show only 3 significant digits
    # stats_format = np.round(stats, 3)  # Round to 3 decimal places
    # stats_format = stats_format.astype(str)  # Convert to string for table
    # row_labels = ["Band 1", "Band 2", "Band 3","Band 4", "Band 5", "Band 6", "Band 7", "Band 8", "Band 9", "Band 10", "Band 11", "Band 12"]
    # col_labels = ["Mean", "Std", "Min", "Max", "Q1", 'Median', 'Q3', 'Skew', 'Kurt']

    # Show statistics as a table
    # fig, ax = plt.subplots()
    # ax.axis("tight")
    # ax.axis("off")

    # Add the table
    # table = ax.table(cellText=stats_format, loc="center", cellLoc="center",rowLabels=row_labels, colLabels=col_labels)
    # table = ax.table(cellText=stats_format, loc="center", cellLoc="center")
    # table.auto_set_font_size(False)
    # table.set_fontsize(7)

    # plt.title("Band Statistics", loc='center', fontsize=16, pad=0.1)
    # plt.show()
    return stats

# ----------- Plot Band Statistics Function----------
def stats_plot(stats,band_names):

    # Format the matrix to show only 3 significant digits
    stats_format = np.round(stats, 3)  # Round to 3 decimal places
    stats_format = stats_format.astype(str)  # Convert to string for table
    col_labels = ["Mean", "Std", "Min", "Max", "Q1", 'Median', 'Q3', 'Skew', 'Kurt']

    # Show statistics as a table
    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")

    # Add the table
    table = ax.table(cellText=stats_format, loc="center", cellLoc="center",rowLabels=band_names, colLabels=col_labels)
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    plt.title("Band Statistics", loc='center', fontsize=16, pad=0.1)
    plt.show()

# ------------------Standardize Function------------------
def standardize(data,wl):

    # Calculate the z score for each pixel
    z_score = np.zeros_like(data)
    for i in range(data.shape[2]):
        z_score[:,:,i] = (data[:,:,i] - np.mean(data[:,:,i])) / np.std(data[:,:,i])

    # Plot histogram of original data
    fig, ax = plt.subplots(3,4,figsize=(12,7))
    axes = ax.flatten()
    for i in range(data.shape[2]):
        axes[i].hist(data[:,:,i].flatten(), bins=256)
        axes[i].set_title(f"{wl[i]}nm ",fontsize=9)
        axes[i].set_xlim(0,1)
    fig.suptitle("Original Data Histograms", fontsize=16)
    plt.tight_layout()
    plt.show()

    # Plot histogram of standardized data
    fig, ax = plt.subplots(3,4,figsize=(12,7))
    axes = ax.flatten()
    for i in range(data.shape[2]):
        axes[i].axvspan(-3, 3, color='black', alpha=0.25)
        axes[i].hist(z_score[:,:,i].flatten(), bins=256,color='orange')
        axes[i].set_title(f"{wl[i]}nm ",fontsize=9)
    fig.suptitle("Standardized Data Histograms", fontsize=16)
    plt.tight_layout()
    plt.show()

# ----------------Correlation Function--------------------
def correlation_matrix(data,wl,stats):

    # reshape data where rows and pixels and cols are bands
    if len(data.shape) > 2:
        reshaped_data = data.reshape(-1, data.shape[2])

    # Calculate the covariance matrix between each band
    cor = np.zeros((reshaped_data.shape[1],reshaped_data.shape[1]))

    for i in range(reshaped_data.shape[1]):
        for j in range(reshaped_data.shape[1]):
            cor[i,j] = ((np.sum((reshaped_data[:,i] - stats[i,0]) * (reshaped_data[:,j] - stats[j,0]))) / reshaped_data.shape[0]) / (stats[i,1]*stats[j,1])

    # Round to 3 decimal places
    cor_format = np.round(cor, 3)

    # Display the array as a heatmap
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.imshow(cor_format, cmap='coolwarm',vmin=-1, vmax=1)

    # Add a colorbar
    plt.colorbar(cax)

    # Set the tick labels
    # tick_spacing = 2
    # ax.set_xticks(wl[::tick_spacing])
    # ax.set_yticks(wl[::tick_spacing])
    # ax.set_xticklabels(wl[::tick_spacing], rotation=45)
    # ax.set_yticklabels(wl[::tick_spacing], rotation=45)

    # Set evenly spaced tick locations
    # Dynamically adjust the number of ticks based on the matrix size
    max_ticks = 15  # Maximum number of ticks to display
    num_bands = cor_format.shape[0]  # Number of spectral bands

    # Ensure we don't set more ticks than available bands
    num_ticks = min(max_ticks, num_bands)

    # Generate tick positions dynamically
    tick_positions = np.linspace(0, num_bands - 1, num=num_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    # Set labels corresponding to selected tick positions
    ax.set_xticklabels([f"{wl[i]:.0f}" for i in tick_positions], rotation=45)
    ax.set_yticklabels([f"{wl[i]:.0f}" for i in tick_positions], rotation=45)



    plt.title("Correlation Matrix")
    plt.xlabel("Bands [nm]")
    plt.ylabel("Bands [nm]")
    plt.show()

    return cor

# -------------Correlation Subplot Function---------------
def correlation_plot(data):

    # Shape the data so rows are pixels and cols are bands
    reshaped_data = data.reshape(-1,data.shape[2])

    # index of 10m bands
    bands_10m = [1,2,3,7]

    # create unique pairs of all the 10m bands
    pairs = list(combinations(bands_10m,2))

    # regular correlation scatter plots
    fig, ax = plt.subplots(2,3,figsize=(10,5))
    axes = ax.flatten()
    for i in range(len(pairs)):
        axes[i].scatter(reshaped_data[:,pairs[i][0]], reshaped_data[:,pairs[i][1]],s=1)
        axes[i].set_title(f"Bands {pairs[i][0]+1} and {pairs[i][1]+1} ")
        axes[i].set_xlabel(f"Band {pairs[i][0]+1}")
        axes[i].set_ylabel(f"Band {pairs[i][1]+1}")
    plt.suptitle("Correlation Plots (10m Bands)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # density correlation scatter plots
    fig, ax = plt.subplots(2,3,figsize=(10,5))
    axes = ax.flatten()

    num_rows_to_select = round(reshaped_data.shape[0] / 100)
    random_indices = np.random.choice(reshaped_data.shape[0], size=num_rows_to_select, replace=False)
    # Use the indices to select the rows
    downsampled_data = reshaped_data[random_indices]
    print("Downsampled Data:")
    print(downsampled_data.shape)

    for i in range(len(pairs)):

        xy = np.vstack([downsampled_data[:, pairs[i][0]], downsampled_data[:, pairs[i][1]]])
        z = gaussian_kde(xy)(xy)
        axes[i].scatter(downsampled_data[:,pairs[i][0]], downsampled_data[:,pairs[i][1]],s=1,c=z)
        axes[i].set_title(f"Bands {pairs[i][0]+1} and {pairs[i][1]+1} ")
        axes[i].set_xlabel(f"Band {pairs[i][0]+1}")
        axes[i].set_ylabel(f"Band {pairs[i][1]+1}")

    plt.suptitle("Correlation Plot Density (10m Bands)", fontsize=16)
    plt.tight_layout()
    plt.show()

# ------------ Spectral Angle Mapper Function-------------
def sam(v1,v2):
    wl = [
        # 442.7,  # Band 1
        492.7,  # Band 2
        559.8,  # Band 3
        664.6,  # Band 4
        704.1,  # Band 5
        740.5,  # Band 6
        782.8,  # Band 7
        832.8,  # Band 8
        864.7,  # Band 8a
        # 945.1,  # Band 9
        # 1373.5, # Band 10
        1613.7,  # Band 11
        2202.4  # Band 12
    ]
    rows, cols, chans = v2.shape
    # Shape the data so rows are bands and cols are pixels
    v2 = v2.reshape(-1,v2.shape[2]).T

    # remove bands affected by absorption
    v2 = np.delete(v2,[0,9],axis=0)
    v1 = np.delete(v1,[0,9],axis=0)

    # calculate the magnitude of target and image vectors
    v1norm = np.linalg.norm(v1)
    v2norm = np.linalg.norm(v2,axis=0)

    # replace the zeros with eps to avoid divide by zero error
    v2norm = np.where(v2norm == 0, np.finfo(float).eps, v2norm)

    # calculate the vector angle in degrees between each pixel and the target
    theta = np.degrees(np.arccos(np.dot(v1,v2) / (v1norm * v2norm)))

    # get sort index of ascending angles
    sort_idx= np.argsort(theta)

    # sort the image pixels accordingly
    v2_sorted = v2[:,sort_idx]

    # take the 100 closest pixels to the target
    v2_match = v2_sorted[:,0:100]

    # plot 1st,5th,100th closest matches
    plt.plot(wl,v1,label="Target")
    plt.plot(wl,v2_match[:,0],label="1st closest")
    plt.plot(wl,v2_match[:,6],label="5th closest")
    plt.plot(wl,v2_match[:,99],label="100th closest")

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Reflectance")
    plt.title("SAM Detection Results")
    plt.legend(loc="upper right")
    plt.show()

    # show detection map at 5 degree threshold
    theta_img = np.reshape(theta,(rows,cols))
    mask = (theta_img <= 5).astype(int)
    plt.imshow(mask)
    plt.colorbar()
    plt.suptitle("SAM Detection Results")
    plt.axis('off')
    plt.show()

    return theta




#---------------------Main Code---------------------------

if __name__ == "__main__":

    # Load Sentinel 2 data
    data = np.load('sentinel2_rochester.npy')
    # Central wavelengths for Sentinel-2A (in nm)
    wl = [
        442.7,  # Band 1
        492.7,  # Band 2
        559.8,  # Band 3
        664.6,  # Band 4
        704.1,  # Band 5
        740.5,  # Band 6
        782.8,  # Band 7
        832.8,  # Band 8
        864.7,  # Band 8a
        945.1,  # Band 9
        #1373.5, # Band 10
        1613.7, # Band 11
        2202.4  # Band 12
    ]

    # Load Oak (Quercus genus) data
    oak_full = np.loadtxt('vegetation.tree.quercus.agrifolia.vswir.jpl108.jpl.asd.spectrum.txt',skiprows=21)
    # downsample oak data
    oak = (np.interp(wl,oak_full[:,0]*1000,oak_full[:,1]))/100

    # Load road
    road_full = np.loadtxt('manmade.road.pavingasphalt.solid.all.0674uuuasp.jhu.becknic.spectrum.txt',skiprows=21)
    # downsample road data
    road = (np.interp(wl,road_full[:,0]*1000,road_full[:,1]))/100

    # -----------------Pre-Processing-----------------------
    # Print the size of the data
    print("size of data",data.shape)

    # Calculate angle needed for rotation
    last_row = np.flip(data[-1, :, 10])
    for i in range(last_row.shape[0]):
        if last_row[i] > 0:
            x = i
            break

    last_col = np.flip(data[:, -1, 10])
    for i in range(last_col.shape[0]):
        if last_col[i] > 0:
            y = i
            break

    theta = np.degrees(np.arctan(y / x))

    #  rotate image
    data_new = rotate(data, -theta)

    # crop image
    data_new = data_new[30:-30, 40:-40]

    # Run the plot band function
    plot_band()

    # Run band statistics function
    stats = calculate_band_statistics()

    # Run standardize function
    standardize(data_new)

    # Run correlation matrix function
    corr_matrix = correlation_matrix(data_new,stats)

    # Run correlation plot function
    correlation_plot(data_new)

    # Run cosine similarity function
    oak_angle = sam(oak,data_new)
    road_angle = sam(road,data_new)

