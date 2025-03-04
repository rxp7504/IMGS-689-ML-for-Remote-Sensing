import numpy as np
import matplotlib.pyplot as plt
import eda.eda as eda
from pca import principal_component_analysis
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import sklearn.metrics

# Compute linear regression metrics
def compute_regression_metrics(y_true, y_pred):
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    std_residuals = np.std(y_true - y_pred)
    return mae, r2, std_residuals


if __name__ == "__main__":

    # import spectral data
    data = np.load('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW3 - Supervised ML/landis_chlorophyl_regression.npy')
    chlor = np.load('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW3 - Supervised ML/landis_chlorophyl_regression_gt.npy')

    # bands
    wl = np.array([490, 560, 600, 620, 650, 665, 705, 740, 842, 865])
    band_names = ("Blue", "Green", "Yellow", "Orange", "Red 1", "Red 2", "Red Edge 1", "Red Edge 2", "NIR_Broad", "NIR1")

    # ----------------- Perform Exploratory Data Analysis (EDA) -------------------

    # show the statistics of each band
    stats = eda.calculate_band_statistics(data)
    eda.stats_plot(stats, band_names)

    # plot original and standardized histograms
    eda.standardize(data,wl)

    # plot correlation matrix
    cor = eda.correlation_matrix(data,wl,stats)

    # balance of chlorophyll distribution for the samples
    plt.hist(chlor, bins=100,edgecolor='black')
    plt.title("Chlorophyll Content Distribution in Training Set")
    plt.xlabel("Chlorophyll Amount")
    plt.ylabel("Frequency")
    plt.show()


    # ----------------------Linear Regression--------------------------

    # standardize data
    # data = (data - np.mean(data,0)) / np.std(data,0)

    # split data into training and testing sets (80% training and 20% testing)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, chlor, test_size=0.2, random_state=42)

    # show target variable distribution for training and testing
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist(y_train, bins=50,edgecolor='black')
    ax[0].set_title("Training Set")
    ax[1].hist(y_test, bins=50,color='g',edgecolor='black')
    ax[1].set_title("Test Set")
    plt.suptitle("Chlorophyll Content Distribution in Training and Test Set")
    plt.show()

    # conduct principal component analysis
    # pcs, eigenvalues, mean_arr = principal_component_analysis(data)
    #
    # # extract PCs
    # pcs_sub = pcs[:,0:5]
    #
    # # rotate the data into PC space
    # data_rot = data @ pcs_sub
    #
    # # split data into training and testing sets (80% training and 20% testing)
    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data_rot, chlor, test_size=0.2, random_state=42)


    # Train the model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Make predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # ridge = Ridge(alpha=.1)  # Try different alpha values (e.g., 0.1, 10)
    # ridge.fit(x_train, y_train)
    # y_train_pred = ridge.predict(x_train)
    # y_test_pred = ridge.predict(x_test)

    # compute regression metrics
    mae_train, r2_train, std_residuals_train = compute_regression_metrics(y_train, y_train_pred) # training metrics
    mae_test, r2_test, std_residuals_test = compute_regression_metrics(y_test, y_test_pred) # testing metrics

    # Print results
    print("\nTraining Set Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae_train:.4f}")
    print(f"  R-squared (R^2): {r2_train:.4f}")
    print(f"  Standard Deviation of Residuals: {std_residuals_train:.4f}")

    print("\nTesting Set Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae_test:.4f}")
    print(f"  R-squared (R^2): {r2_test:.4f}")
    print(f"  Standard Deviation of Residuals: {std_residuals_test:.4f}")

    # regression plots
    # Plot actual vs predicted for test set
    plt.scatter(y_train, y_train_pred, color="blue", alpha=0.6, label="Predicted vs Actual")
    plt.plot([chlor.min(), chlor.max()], [chlor.min(), chlor.max()], "k--", lw=2)  # Identity line
    plt.xlabel("Actual Chlorophyll Content")
    plt.ylabel("Predicted Chlorophyll Content")
    plt.title("Linear Regression - Prediction vs Actual")
    plt.legend()
    plt.show()

    import pandas as pd
    import seaborn as sns

    df = pd.DataFrame(data, columns=["Band_" + str(w) for w in wl])  # Feature names
    df["Chlorophyll"] = chlor  # Target variable

    feature_target_correlation = df.corr()["Chlorophyll"].drop("Chlorophyll")

    # Plot correlation with target
    sns.barplot(x=feature_target_correlation.index, y=feature_target_correlation.values)
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.title("Feature-Target Correlation")
    plt.show()

    plt.scatter(data[:,6],chlor,edgecolors='black')
    plt.title(f'Red-Edge Band ({wl[6]} nm) vs Chlorophyll Content')
    plt.xlabel('Red-Edge Band Value')
    plt.ylabel('Chlorophyll')
    plt.show()

