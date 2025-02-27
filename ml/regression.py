import numpy as np
import matplotlib.pyplot as plt
import eda.eda as eda




if __name__ == "__main__":

    # import spectral data
    data = np.load('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW3 - Supervised ML/landis_chlorphyl_regression.npy')
    chlor = np.load('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW3 - Supervised ML/landis_chlorphyl_regression_gt.npy')

    # bands
    wl = np.array([490, 560, 600, 620, 650, 665, 705, 740, 842, 865])
    band_names = ("Blue", "Green", "Yellow", "Orange", "Red 1", "Red 2", "Red Edge 1", "Red Edge 2", "NIR_Broad", "NIR1")

    # ----------------- Perform Exploratory Data Analysis (EDA) -------------------
    stats = eda.calculate_band_statistics(data)
    eda.stats_plot(stats, band_names)


p