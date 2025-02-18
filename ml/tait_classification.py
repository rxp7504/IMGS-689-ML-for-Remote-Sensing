import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from spectral import envi
from pca import principal_component_analysis
from pca import display_rgb
import kmeans as km
from sklearn.cluster import MiniBatchKMeans

# load the envi image
img = envi.open('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive'
                '/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW2 - PCA and K-means/'
                'materials/tait_hsi.hdr')
data = img.load()
data = np.array(data, copy=True)

# load the band centers and convert to a numerical vector
band_names = img.metadata.get('band names', None)
wl = np.array([float(w.split()[0]) for w in band_names])

# remove bad bands
badbands = 272
data = np.delete(data, badbands, 2)
wl = np.delete(wl, badbands)

print(img)

# reshape data
data_reshaped = data.reshape(-1, data.shape[2])

# Define number of clusters
K = 8

# initialize and fit MiniBatchKMeans for original data
kmeans = MiniBatchKMeans(n_clusters=K, batch_size=250, random_state=42)
kmeans.fit(data_reshaped)

# Get class labels and reshape back to image dimensions
map = kmeans.labels_.reshape(data.shape[0], data.shape[1])
rgb = display_rgb(data,wl)

# display
fig, ax = plt.subplots(1,2,figsize=(8,3))
ax[0].imshow(rgb)
ax[0].axis('off')
ax[0].set_title('RGB Image')
ax[1].imshow(map, cmap='Set1')
ax[1].axis('off')
ax[1].set_title(f'Mini-Batch KMeans - Original Data ({K} Classes)')
plt.show()

# pca calculation
pcs, eigenvalues, mean_arr = principal_component_analysis(data_reshaped)

# Perform PCA, using 2, 5, 10, 50, and 100 features
features = [2,5,10,50,100]
fig, ax = plt.subplots(1, len(features),figsize=(10,3))

for i in range(len(features)):

    pcs_sub = pcs[:,0:features[i]]

    # rotate the data into PC space
    data_pca = data_reshaped @ pcs_sub

    # initialize and fit MiniBatchKMeans for pca data
    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=250, random_state=42)
    kmeans.fit(data_pca)

    # Get class labels and reshape back to image dimensions
    map = kmeans.labels_.reshape(data.shape[0], data.shape[1])

    ax[i].imshow(map, cmap='Set1')
    ax[i].axis('off')
    ax[i].set_title(f'{features[i]} PCs')

plt.suptitle(f'Kmeans with Varying PCs (K = {K})')
plt.show()
