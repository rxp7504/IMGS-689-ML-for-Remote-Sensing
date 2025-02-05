import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
from spectral import envi

# load the envi image
img = envi.open('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive'
                       '/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW2 - PCA and K-means/'
                       'materials/tait_hsi.hdr')
data = img.load()
print(img)
print(data.shape)

# load the band centers and convert to a numerical vector
band_names = img.metadata.get('band names',None)
wl = np.array([float(w.split()[0]) for w in band_names])

