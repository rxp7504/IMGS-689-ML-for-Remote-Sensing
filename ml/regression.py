import numpy as np
import matplotlib.pyplot as plt
import eda.eda as eda
from pca import principal_component_analysis
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import sklearn.metrics

# -------Function to compute linear regression metrics--------
def compute_regression_metrics(y_true, y_pred):
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    std_residuals = np.std(y_true - y_pred)
    return mae, r2, std_residuals

# -------Function to compute feature/target correlation--------
def feature_target_correlation(data,target,wl):
    # reshape data where rows and pixels and cols are bands
    if len(data.shape) > 2:
        reshaped_data = data.reshape(-1, data.shape[2])
    target = target.reshape(-1, 1)
    r = np.sum((data - data.mean(axis=0)) * (target.reshape(-1, 1) - target.mean()), axis=0) / \
        np.sqrt(np.sum((data - data.mean(axis=0)) ** 2, axis=0) * np.sum((target - target.mean()) ** 2))
    return r


if __name__ == "__main__":

    # import spectral data
    data = np.load('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW3 - Supervised ML/landis_chlorophyl_regression.npy')
    chlor = np.load('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW3 - Supervised ML/landis_chlorophyl_regression_gt.npy')

    # bands
    wl = np.array([490, 560, 600, 620, 650, 665, 705, 740, 842, 865])
    band_names = ("Blue", "Green", "Yellow", "Orange", "Red 1", "Red 2", "Red Edge 1", "Red Edge 2", "NIR_Broad", "NIR1")
    wl_labels = [str(w) for w in wl]

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

    # plot correlation with target
    r = feature_target_correlation(data,chlor,wl)
    plt.bar(wl_labels, r,edgecolor='black')
    plt.title("Feature-Target Correlation")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Correlation")
    plt.show()

    # plot relationship of each feature with target
    fig, ax = plt.subplots(2,5,figsize=(10,5))
    axes = ax.flatten()  # Flatten 2D array to 1D for easy indexing
    for i in range(wl.shape[0]):
        axes[i].scatter(data[:,i],chlor,edgecolors='black')
        axes[i].set_title(f'{wl[i]} nm')
        axes[i].set_xlabel('Band Value')
        axes[i].set_ylabel('Chlorophyll')
    plt.suptitle('Relationship of Chlorophyll Content Per Band')
    plt.tight_layout()
    plt.show()


    # ----------------------Linear Regression--------------------------

    # trim irrelevant features
    # data = data[:,0:7]

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

    # train model using ridge regression to prevent effects from multicolinearity
    ridge = Ridge(alpha=10)  # Try different alpha values (e.g., 0.1, 10)
    ridge.fit(x_train, y_train)
    y_train_pred = ridge.predict(x_train)
    y_test_pred = ridge.predict(x_test)

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

    # Regression Plots
    fig,ax = plt.subplots(1,2,figsize=(10,5))

    ax[0].scatter(y_train, y_train_pred, color="blue", alpha=0.6, label="Predicted Chlorophyll")
    ax[0].plot([chlor.min(), chlor.max()], [chlor.min(), chlor.max()], "k--", lw=2)  # Identity line
    ax[0].set_xlabel("Actual Chlorophyll Content")
    ax[0].set_ylabel("Predicted Chlorophyll Content")
    ax[0].set_title(f"Training Set")

    # Display metrics on Training Set plot
    train_metrics = f"MAE: {mae_train:.4f}\nR²: {r2_train:.4f}\nStd Res: {std_residuals_train:.4f}"
    ax[0].text(0.05, 0.95, train_metrics, transform=ax[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax[1].scatter(y_test, y_test_pred, color="green", alpha=0.6, label="Predicted Chlorophyll")
    ax[1].plot([chlor.min(), chlor.max()], [chlor.min(), chlor.max()], "k--", lw=2)  # Identity line
    ax[1].set_xlabel("Actual Chlorophyll Content")
    ax[1].set_ylabel("Predicted Chlorophyll Content")
    ax[1].set_title(f"Test Set")

    # Display metrics on Test Set plot
    test_metrics = f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Res: {std_residuals_test:.4f}"
    ax[1].text(0.05, 0.95, test_metrics, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.suptitle('Regression Plots',fontweight='bold')
    plt.show()

    # Residual Plots
    fig,ax = plt.subplots(1,2,figsize=(10,5))

    ax[0].scatter(y_train_pred, y_train - y_train_pred, color="blue", alpha=0.6, label="Predicted Chlorophyll")
    ax[0].axhline(0, color="k", linestyle="--", lw=2)  # Horizontal line at y=0
    ax[0].set_ylabel("Residuals")
    ax[0].set_xlabel("Predicted Chlorophyll Content")
    ax[0].set_title(f"Training Set")
    ax[0].set_ylim(-20,20)
    # Display metrics on Training Set plot
    train_metrics = f"MAE: {mae_train:.4f}\nR²: {r2_train:.4f}\nStd Res: {std_residuals_train:.4f}"
    ax[0].text(0.05, 0.95, train_metrics, transform=ax[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


    ax[1].scatter(y_test_pred, y_test - y_test_pred, color="green", alpha=0.6, label="Predicted Chlorophyll")
    ax[1].axhline(0, color="k", linestyle="--", lw=2)  # Horizontal line at y=0
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlabel("Predicted Chlorophyll Content")
    ax[1].set_title(f"Test Set")
    ax[1].set_ylim(-20,20)
    # Display metrics on Test Set plot
    test_metrics = f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Res: {std_residuals_test:.4f}"
    ax[1].text(0.05, 0.95, test_metrics, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


    plt.suptitle('Residual Plots',fontweight='bold')
    plt.show()

    # ----------------------Partial Least Squares Regression (PLSR):--------------------------



