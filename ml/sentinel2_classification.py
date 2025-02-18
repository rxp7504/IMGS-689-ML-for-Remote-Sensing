import numpy as np
import matplotlib.pyplot as plt
import kmeans as km
import pca as pca

if __name__ == "__main__":


    # Load Sentinel 2 data
    data = np.load('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW 1 - EDA/Python Code/sentinel2_rochester.npy')
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
        # 1373.5, # Band 10
        1613.7,  # Band 11
        2202.4  # Band 12
    ]
    print("Data Shape:",data.shape)

    # display rgb of sentinel 2
    pca.display_rgb(data,wl)

    # remove invalid pixels
    data_nopad,pad_idx = pca.remove_padding(data)

    # pca calculation
    pcs, eigenvalues, mean_arr = pca.principal_component_analysis(data_nopad)

    # extract first 3,4,5 and 6 PCs
    vals = [3,4,5,6]
    fig, ax = plt.subplots(1, len(vals),figsize=(10,5))
    for i in range(len(vals)):
        pcs_sub = pcs[:,0:vals[i]]

        # rotate the data into PC space
        data_rot = data_nopad @ pcs_sub

        # reshape data to image dimensions
        data_pca = pca.add_padding(data,data_rot,pad_idx)
        print(data_pca.min(),data_pca.max())

        # apply kmeans clustering to reduced data
        map = km.kmeans(data_pca,5,100)
        ax[i].imshow(map,cmap='Set1')
        ax[i].axis('off')
        ax[i].set_title(f'{vals[i]} PCs')

    plt.suptitle('Kmeans with Varying PCs')
    plt.show()


