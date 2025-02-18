import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
from numpy.ma.core import masked_values, zeros_like
from spectral import envi
from spectral import open_image
import eda.eda as eda
from scipy.ndimage import rotate
import matplotlib
# matplotlib.use('TkAgg')



def rescale(arr):
# Function for scaling an image 0-1 for visualization

    arr = np.asarray(arr)  # ensure input is numpy array
    # rescale the values from the minimum nonzero value to one and clamp
    dst = np.clip((arr - np.min(arr[arr>0])) / (np.max(arr[arr>0]) - np.nanmin(arr[arr>0])),0,1)
    return dst

def principal_component_analysis(array):

    if len(array.shape) == 3:
        # reshape data so rows are pixels and cols are bands
        array = array.reshape(-1, array.shape[2])

    # mean center each band
    array_centered = array - np.mean(array, axis=0)

    # calculate covariance matrix
    cov_mat = (array_centered.T @ array_centered) / (array_centered.shape[0])

    # conduct singular value decomposition
    U, S, Vt = np.linalg.svd(cov_mat)
    pcs = U
    eigenvalues = S
    mean_arr = array_centered
    print("Eigenvalues: ",eigenvalues.shape)
    print("Eigen Vectors: ",pcs.shape)

    return pcs, eigenvalues, mean_arr

def select_pixels(img):

    # Display the image
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.title("Click 5 pixel locations")
    plt.xlabel("X-axis (columns)")
    plt.ylabel("Y-axis (rows)")

    # Allow the user to select 5 points
    coords = plt.ginput(5)

    # Print selected coordinates
    print("Selected pixels:", coords)

    plt.show()
    return coords

def impact_plot(data,eigenvalues,pcs,b):
    k = 2
    # mean of each band
    x_bar = np.mean(data, axis=0)

    # calculate positive impact
    pos_impact = ((k * np.sqrt(eigenvalues)) * pcs)

    # calculate negative impact
    neg_impact = ((-k * np.sqrt(eigenvalues)) * pcs)

    fig, ax = plt.subplots(1,b,figsize = (5*b,5))
    for i in range(b):
        #plot
        ax[i].plot(x_bar-x_bar,label = "Mean")
        ax[i].plot(pos_impact[:,i],label = "Positive Impact")
        ax[i].plot(neg_impact[:,i],label = "Negative Impact")
        ax[i].set_xlabel("Spectral Bands")
        ax[i].set_ylabel("Reflectance")
        ax[i].set_title(f"Impact Plot for PC {i+1}")

    plt.legend()
    plt.suptitle("Impact Plots",fontsize = 12,fontweight="bold")
    plt.tight_layout()
    plt.show()

def display_rgb(data,wl):
    wl = np.array(wl)  # Ensure wl is a NumPy array
    # plot bands from red, green and blue regions of the spectrum
    idx_b = np.argmin(abs(wl - 450))
    idx_g = np.argmin(abs(wl - 550))
    idx_r = np.argmin(abs(wl - 650))

    r_band = rescale(data[:,:,idx_r])
    g_band = rescale(data[:,:,idx_g])
    b_band = rescale(data[:,:,idx_b])
    rgb_band = np.dstack((r_band,g_band,b_band))

    # just plot the rgb image alone
    fig = plt.figure(facecolor='white')
    plt.imshow(rgb_band)
    plt.axis('off')
    plt.show()
    return rgb_band

def remove_padding(data):
    # reshape data so rows are pixels and cols are bands
    data_reshaped = data.reshape(-1,data.shape[2])

    # find locations of invalid pixels
    data_mean = np.mean(data, axis=(0, 1))
    pad_idx = data_reshaped[:,np.argmax(data_mean)] == 0

    # remove invalid pixels for calculations
    data_nopad = data_reshaped[~pad_idx,:]
    return data_nopad,pad_idx

def add_padding(data,data_rot,pad_idx):
    # reshape data so rows are pixels and cols are bands
    data_reshaped = data.reshape(-1,data.shape[2])

    # add zero padding back in
    data_rotpad = np.zeros((data_reshaped.shape[0],data_rot.shape[1]))
    data_rotpad[~pad_idx,:] = data_rot

    # reshape to original image dimensions
    data_pca = data_rotpad.reshape(data.shape[0],data.shape[1],data_rotpad.shape[1])

    return data_pca

if __name__ == "__main__":

    # load the envi image
    img = envi.open('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive'
                           '/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW2 - PCA and K-means/'
                           'materials/tait_hsi.hdr')
    data = img.load()
    data = np.array(data, copy=True)

    # load the band centers and convert to a numerical vector
    band_names = img.metadata.get('band names',None)
    wl = np.array([float(w.split()[0]) for w in band_names])

    # remove bad bands
    badbands = 272
    data = np.delete(data, badbands, 2)
    wl = np.delete(wl, badbands)

    print(img)
    print("Data Shape: ",data.shape)

    # plot bands from red, green and blue regions of the spectrum
    idx_b = np.argmin(abs(wl - 450))
    idx_g = np.argmin(abs(wl - 550))
    idx_r = np.argmin(abs(wl - 650))

    r_band = rescale(data[:,:,idx_r])
    g_band = rescale(data[:,:,idx_g])
    b_band = rescale(data[:,:,idx_b])
    rgb_band = np.dstack((r_band,g_band,b_band))

    fig, axes = plt.subplots(1,4,figsize=(15,5))
    axes[0].imshow(r_band, cmap='gray')
    axes[0].set_title(f'Red Band ({wl[idx_r]}nm)')
    axes[0].axis('off')

    axes[1].imshow(g_band, cmap='gray')
    axes[1].set_title(f'Green Band ({wl[idx_g]}nm)')
    axes[1].axis('off')

    axes[2].imshow(b_band, cmap='gray')
    axes[2].set_title(f'Blue Band ({wl[idx_b]}nm)')
    axes[2].axis('off')

    axes[3].imshow(rgb_band, cmap='gray')
    axes[3].set_title('RGB Band (combined)')
    axes[3].axis('off')
    plt.show()

    # just plot the rgb image alone
    fig = plt.figure(facecolor='white')
    plt.imshow(rgb_band)
    plt.axis('off')
    plt.show()

    # create pseudocolor image from green, red, and NIR bands
    idx_nir = np.argmin(abs(wl - 950))
    nir_band = rescale(data[:,:,idx_nir])
    nirRG_bands = np.dstack((nir_band, r_band, g_band))

    fig = plt.figure(facecolor='white')
    plt.imshow(nirRG_bands, cmap='grey')
    plt.title('NIR-R-G Pseudocolor Image')
    plt.axis('off')
    plt.show()


    # reshape data so rows are pixels and cols are bands
    data_reshaped = data.reshape(-1,data.shape[2])

    # find locations of invalid pixels
    data_mean = np.mean(data, axis=(0, 1))
    pad_idx = data_reshaped[:,np.argmax(data_mean)] == 0

    # remove invalid pixels for calculations
    data_nopad = data_reshaped[~pad_idx,:]

    k = 10
    # calculate statistics of each band
    stats = eda.calculate_band_statistics(data_nopad[:, 0:k])

    # calculate and display the correlation matrix
    cor = eda.correlation_matrix(data_nopad[:,0:k],stats)

    # pca calculation
    pcs, eigenvalues, mean_arr = principal_component_analysis(data_nopad)

    # extract first 10 PCs
    pcs10 = pcs[:,0:10]

    # rotate the data into PC space
    data_rot = data_nopad @ pcs10

    # add zero padding back in
    data_rotpad = np.zeros((data_reshaped.shape[0],10))
    data_rotpad[~pad_idx,:] = data_rot

    # reshape to original image dimensions
    data_pca = data_rotpad.reshape(data.shape[0],data.shape[1],data_rotpad.shape[1])
    print(data_pca.shape)

    # plot the first 10 PCs with a variance label
    fig, axes = plt.subplots(2,5,figsize=(9,6))
    axes = axes.flatten()
    for i in range(data_pca.shape[2]):
        axes[i].imshow((data_pca[:,:,i]), cmap='gray')
        axes[i].set_title(f"PC {i + 1} (σ²: {np.var(data_rot[:, i]):.1e})")
        axes[i].axis('off')

    plt.tight_layout()
    plt.suptitle('Principal Components',fontsize=16,fontweight='bold')
    plt.show()

    # plot eigenvalues/variances
    plt.plot(eigenvalues)
    plt.yscale('log')
    plt.ylabel('Eigenvalues')
    plt.xlabel('PC Bands')
    plt.title('Principal Components Eigenvalues')
    plt.show()

    # mean center each band
    data_centered = data_nopad - np.mean(data_nopad, axis=0)

    # number of pcs
    pcs_subset = (1,10,50,100,data.shape[2])

    # calculate the mean reconstruction error
    error = np.zeros(len(pcs_subset)) # initialize error
    for i in range(len(pcs_subset)):
        pcs_sub = pcs[:, :pcs_subset[i]] # pc subset
        data_rot_sub = data_centered @ pcs_sub # project data into reduced space
        data_reconstructed = data_rot_sub @ pcs_sub.T # reconstruct the data
        error[i] = np.mean(np.linalg.norm(data_reconstructed - data_centered,axis=1)) # calculate L2 distance (reconstruction error)

    # plot the error
    plt.scatter(pcs_subset,error)
    plt.xlabel('Number of PCs')
    plt.ylabel('Error')
    plt.title('Reconstruction Error')
    plt.show()

    # compute explained variance ratio
    evr = np.zeros(eigenvalues.shape)
    for i in range(eigenvalues.shape[0]):
        evr[i] = eigenvalues[i]/np.sum(eigenvalues)
        if np.sum(evr) >= 0.99: # take idx of pcs with at least 99% of cumulative variance
            evr_maxidx = i
            break

    # set remaining pcs to 0
    pcs_nr = np.zeros_like(pcs)
    pcs_nr[:,0:evr_maxidx] = pcs[:,0:evr_maxidx]

    # reconstruct the noise reduced data
    data_reconstructed = np.clip((data_rot_sub @ pcs_nr.T) + np.mean(data_nopad, axis=0),0,None) # add mean and clamp negative values
    data_reconstructed_pad = np.zeros_like(data_reshaped)
    data_reconstructed_pad[~pad_idx,:] = data_reconstructed # add padding back in for display

    # reshape to original image dimensions
    data_reconstructed_pad = data_reconstructed_pad.reshape(data.shape[0],data.shape[1],data.shape[2])

    # display image of before and after for the first band
    k = 0
    fig,axes = plt.subplots(1,2,figsize=(6.4,3))
    axes[0].imshow(rescale(data[:,:,k]), cmap='gray')
    axes[0].set_title(f'Original band {k}')
    axes[0].axis('off')
    axes[1].imshow(rescale(data_reconstructed_pad[:,:,k]), cmap='gray')
    axes[1].axis('off')
    axes[1].set_title(f'Noise reduced band {k}')
    plt.show()
    print("min data value: ", np.min(data_reconstructed))
    print("max data value: ", np.max(data_reconstructed_pad[:,:,idx_r]))

    # select 5 pixels to compare
    coords = np.round(np.array([(392.70887445887433, 84.95887445887433), (603.5449134199133, 706.2224025974026), (760.9691558441558, 472.8971861471862), (679.4458874458874, 686.5443722943724), (696.3127705627705, 551.6093073593074)])).astype(int)
    panels = ('Red Panel','Green Panel','Blue Panel','Yellow Panel','Black Panel')

    # plot spectra of pixels before and after noise reduction
    fig,axes = plt.subplots(1,5,figsize=(15,3))
    for i,ax in enumerate(axes):
        ax.plot(wl,data[coords[i,1],coords[i,0],:],label='Original')
        ax.plot(wl,data_reconstructed_pad[coords[i,1],coords[i,0],:],label='Noise Reduced')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Value')
        ax.set_title(panels[i])

    fig.suptitle('PCA Noise Reduction Results',fontsize=16,fontweight='bold')
    fig.legend(['Original','Noise Reduced'])
    plt.tight_layout()
    plt.show()

    # calculate SNR for entire image
    snr_original = np.mean(data_nopad, axis=0)/np.std(data_nopad, axis=0)
    snr_original_img = np.round(np.mean(data_nopad, axis=(0,1))/np.std(data_nopad, axis=(0,1)),3)

    # calculate SNR for noise reduced image
    snr_nr = np.mean(data_reconstructed, axis=0)/np.std(data_reconstructed, axis=0)
    snr_nr_img = np.round(np.mean(data_reconstructed, axis=(0,1))/np.std(data_reconstructed, axis=(0,1)),3)

    # SNR plots
    plt.plot(wl,snr_original)
    plt.plot(wl,snr_nr)
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('SNR')
    plt.title('SNR Noise Reduction Results')
    plt.legend([f'Original: {snr_original_img:.3f}',f'Noise Reduced: {snr_nr_img:.3f}'])
    plt.tight_layout()
    plt.show()

    # impact plots
    impact_plot(data_nopad,eigenvalues,pcs,5)