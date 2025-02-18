import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as imageio

def kmeans(data,K,iterations):

    # reshape data so rows are pixels and cols are bands
    data_reshaped = data.reshape(-1, data.shape[2])
    print("Data Reshaped:",data_reshaped.shape)

    # standardize the data
    data_standardized = (data_reshaped - np.mean(data_reshaped, axis=0)) /np.std(data_reshaped, axis=0)

    # initialize k cluster centers from existing data points
    np.random.seed(42)
    rand_rows = np.random.choice(data_reshaped.shape[0], K, replace=False)
    mu = data_standardized[rand_rows,:]
    print("Starting Centroids:",mu)

    # initialize avg class distance
    error = np.zeros([iterations,K])

    # loop over iterations
    for j in range(iterations):

        # calculate distance between each pixel and each centroid
        l2 = np.zeros([data_standardized.shape[0],K])
        for i in range(K):
            l2[:,i] = np.sqrt(np.sum(np.square(data_standardized - mu[i,:]), axis=1))

        # determine which class each pixel is closest to
        c = np.argmin(l2,axis=1)

        # move each centroid to the mean of the points assigned to it
        for i in range(K):
            mu[i,:] = np.mean(data_standardized[c==i,:], axis=0)
            error[j,i] = np.mean(np.sqrt(np.sum(np.square(data_standardized[c == i, :] - mu[i, :]), axis=1)))

        # break if distances have not changed for 10 iterations
        if np.array_equal(error[j, :], error[j - 10, :]):
            print("Model has converged")
            break

    # # plot the avg distance from each centroid per iteration
    # for i in range(K):
    #     plt.plot(error[:j,i],label=f'Class {i+1}')

    # plt.title('Average Class Distance Per Iteration')
    # plt.ylabel('Average Class Distance')
    # plt.xlabel('Iteration')
    # plt.legend(loc='upper right')
    # plt.show()
    # plt.show()

    # class prediction map
    map = c.reshape(data.shape[0],data.shape[1])

    return map

if __name__ == "__main__":

    # load image
    data = imageio.imread('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW2 - PCA and K-means/materials/jellybeans.tiff')

    K = 6
    class_map = kmeans(data,K,100)

    fig,ax = plt.subplots(1,2,figsize=(6,3))
    ax[0].imshow(data)
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    ax[1].imshow(class_map, cmap='Set1')
    ax[1].axis('off')
    ax[1].set_title(f'Class Predictions ({K} classes)')
    plt.show()