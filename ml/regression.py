import numpy as np
import matplotlib.pyplot as plt
import eda.eda as eda
import sklearn.model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression
import sklearn.metrics
from sklearn.neural_network import MLPRegressor


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
    # standardize data
    data = (data - data.mean(axis=0))/data.std(axis=0)

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
    print('\n----------------LINEAR REGRESSION (RIDGE)-----------------')
    print("Training Set Metrics:")
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
    ax[0].plot([chlor.min(), chlor.max()], [chlor.min(), chlor.max()], "k--", lw=2,label='1:1 line')  # Identity line
    ax[0].set_xlabel("Actual Chlorophyll Content")
    ax[0].set_ylabel("Predicted Chlorophyll Content")
    ax[0].set_title(f"Training Set")
    ax[0].grid(True)
    ax[0].legend()
    # Display metrics on Training Set plot
    train_metrics = f"MAE: {mae_train:.4f}\nR²: {r2_train:.4f}\nStd Res: {std_residuals_train:.4f}"
    ax[0].text(0.05, 0.95, train_metrics, transform=ax[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax[1].scatter(y_test, y_test_pred, color="green", alpha=0.6, label="Predicted Chlorophyll")
    ax[1].plot([chlor.min(), chlor.max()], [chlor.min(), chlor.max()], "k--", lw=2,label='1:1 line')  # Identity line
    ax[1].set_xlabel("Actual Chlorophyll Content")
    ax[1].set_ylabel("Predicted Chlorophyll Content")
    ax[1].set_title(f"Test Set")
    ax[1].grid(True)
    ax[1].legend()
    # Display metrics on Test Set plot
    test_metrics = f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Res: {std_residuals_test:.4f}"
    ax[1].text(0.05, 0.95, test_metrics, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.suptitle('Regression Plots',fontweight='bold')
    plt.show()

    # Residual Plots
    fig,ax = plt.subplots(1,2,figsize=(10,5))

    ax[0].scatter(y_train_pred, y_train - y_train_pred, color="blue", alpha=0.6, label="Prediction Residuals")
    ax[0].axhline(0, color="k", linestyle="--", lw=2,label='0 line')  # Horizontal line at y=0
    ax[0].set_ylabel("Residuals")
    ax[0].set_xlabel("Predicted Chlorophyll Content")
    ax[0].set_title(f"Training Set")
    ax[0].set_ylim(-20,20)
    ax[0].grid(True)
    ax[0].legend()
    # Display metrics on Training Set plot
    train_metrics = f"MAE: {mae_train:.4f}\nR²: {r2_train:.4f}\nStd Res: {std_residuals_train:.4f}"
    ax[0].text(0.05, 0.95, train_metrics, transform=ax[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


    ax[1].scatter(y_test_pred, y_test - y_test_pred, color="green", alpha=0.6, label="Prediction Residuals")
    ax[1].axhline(0, color="k", linestyle="--", lw=2,label='0 line')  # Horizontal line at y=0
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlabel("Predicted Chlorophyll Content")
    ax[1].set_title(f"Test Set")
    ax[1].set_ylim(-20,20)
    ax[1].grid(True)
    ax[1].legend()
    # Display metrics on Test Set plot
    test_metrics = f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Res: {std_residuals_test:.4f}"
    ax[1].text(0.05, 0.95, test_metrics, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


    plt.suptitle('Residual Plots',fontweight='bold')
    plt.show()

    # ----------------------Partial Least Squares Regression (PLSR):--------------------------

    # set the number of components to vary
    numComponents = list(range(1,data.shape[1]+1))
    r2_train_scores = []
    for i in numComponents:
        # train using PLSR
        pls = PLSRegression(n_components=i)
        pls.fit(x_train, y_train)

        # Predict on training set
        y_train_pred = pls.predict(x_train)

        mae, r2, std_residuals = compute_regression_metrics(y_train,y_train_pred)
        r2_train_scores.append(r2)

    # plot the effect of accuracy vs number of components
    plt.plot(numComponents, r2_train_scores,linestyle="-", marker=".")
    plt.axvline(numComponents[np.argmax(r2_train_scores)], color="r", linestyle="--",label="Max Score")
    plt.title("Training Accuracy for Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("R²")
    plt.legend()
    plt.show()

    # train and predict on train and test sets
    pls = PLSRegression(n_components=numComponents[np.argmax(r2_train_scores)]) # best number of components
    pls.fit(x_train, y_train) # train using PLS
    y_train_pred = pls.predict(x_train) # predict using training data
    y_test_pred = pls.predict(x_test) # predict using test data

    # compute regression metrics
    mae_train, r2_train, std_residuals_train = compute_regression_metrics(y_train, y_train_pred) # training metrics
    mae_test, r2_test, std_residuals_test = compute_regression_metrics(y_test, y_test_pred) # testing metrics

    # Regression Plots
    fig,ax = plt.subplots(1,2,figsize=(10,5))

    ax[0].scatter(y_train, y_train_pred, color="blue", alpha=0.6, label="Predicted Chlorophyll")
    ax[0].plot([chlor.min(), chlor.max()], [chlor.min(), chlor.max()], "k--", lw=2,label='1:1 line')  # Identity line
    ax[0].set_xlabel("Actual Chlorophyll Content")
    ax[0].set_ylabel("Predicted Chlorophyll Content")
    ax[0].set_title(f"Training Set")
    ax[0].grid(True)
    ax[0].legend()

    # Display metrics on Training Set plot
    train_metrics = f"MAE: {mae_train:.4f}\nR²: {r2_train:.4f}\nStd Res: {std_residuals_train:.4f}"
    ax[0].text(0.05, 0.95, train_metrics, transform=ax[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    ax[1].scatter(y_test, y_test_pred, color="green", alpha=0.6, label="Predicted Chlorophyll")
    ax[1].plot([chlor.min(), chlor.max()], [chlor.min(), chlor.max()], "k--", lw=2,label='1:1 line')  # Identity line
    ax[1].set_xlabel("Actual Chlorophyll Content")
    ax[1].set_ylabel("Predicted Chlorophyll Content")
    ax[1].set_title(f"Test Set")
    ax[1].grid(True)
    ax[1].legend()
    # Display metrics on Test Set plot
    test_metrics = f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Res: {std_residuals_test:.4f}"
    ax[1].text(0.05, 0.95, test_metrics, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.suptitle('Regression Plots Using PLS',fontweight='bold')
    plt.show()


    # Residual Plots
    fig,ax = plt.subplots(1,2,figsize=(10,5))

    ax[0].scatter(y_train_pred, y_train - y_train_pred, color="blue", alpha=0.6, label="Prediction Residuals")
    ax[0].axhline(0, color="k", linestyle="--", lw=2,label='0 line')  # Horizontal line at y=0
    ax[0].set_ylabel("Residuals")
    ax[0].set_xlabel("Predicted Chlorophyll Content")
    ax[0].set_title(f"Training Set")
    ax[0].set_ylim(-20,20)
    ax[0].grid(True)
    ax[0].legend()
    # Display metrics on Training Set plot
    train_metrics = f"MAE: {mae_train:.4f}\nR²: {r2_train:.4f}\nStd Res: {std_residuals_train:.4f}"
    ax[0].text(0.05, 0.95, train_metrics, transform=ax[0].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))


    ax[1].scatter(y_test_pred, y_test - y_test_pred, color="green", alpha=0.6, label="Prediction Residuals")
    ax[1].axhline(0, color="k", linestyle="--", lw=2,label='0 line')  # Horizontal line at y=0
    ax[1].set_ylabel("Residuals")
    ax[1].set_xlabel("Predicted Chlorophyll Content")
    ax[1].set_title(f"Test Set")
    ax[1].set_ylim(-20,20)
    ax[1].grid(True)
    ax[1].legend()
    # Display metrics on Test Set plot
    test_metrics = f"MAE: {mae_test:.4f}\nR²: {r2_test:.4f}\nStd Res: {std_residuals_test:.4f}"
    ax[1].text(0.05, 0.95, test_metrics, transform=ax[1].transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    plt.suptitle('Residual Plots Using PLS',fontweight='bold')
    plt.show()

    # Print results
    print('\n----------------PARTIAL LEAST SQUARES REGRESSION (PLS)-----------------')
    print("Training Set Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae_train:.4f}")
    print(f"  R-squared (R^2): {r2_train:.4f}")
    print(f"  Standard Deviation of Residuals: {std_residuals_train:.4f}")

    print("\nTesting Set Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae_test:.4f}")
    print(f"  R-squared (R^2): {r2_test:.4f}")
    print(f"  Standard Deviation of Residuals: {std_residuals_test:.4f}")


    # ------------------- REGRESSION USING MULTIPLE LAYER PERCEPTRON (MLP)-------------------

    # define a range of layer sizes each with one neuron
    layerList = [
        (10,),  # 1 hidden layer, 10 neuron
        (10, 10),  # 2 hidden layers, 10 neuron each
        (10, 10, 10),  # 3 hidden layers, 10 neuron each
        (10, 10, 10, 10),  # 4 hidden layers, 10 neuron each
        (10, 10, 10, 10, 10)  # 5 hidden layers, 10 neuron each
    ]
    r2_train_scores = []

    fig1, ax1 = plt.subplots(1, len(layerList), figsize=(20, 5))
    fig2, ax2 = plt.subplots(1, len(layerList), figsize=(20, 5))


    for i,layer in enumerate(layerList):
        # initialize MLP regressor
        mlp = MLPRegressor(hidden_layer_sizes=layerList[i],activation='relu',max_iter=5000,random_state=42,learning_rate_init=0.001)

        # train the MLP
        mlp.fit(x_train, y_train)

        # predict on training set
        y_pred = mlp.predict(x_train)

        mae, r2, std_residuals = compute_regression_metrics(y_train, y_pred)

        # predict on test set
        y_pred = mlp.predict(x_test)
        mae, r2, std_residuals = compute_regression_metrics(y_test, y_pred)
        r2_train_scores.append(r2)

        # regression plot per model
        ax1[i].scatter(y_test, y_pred, color="green", alpha=0.6, label="Predicted Chlorophyll")
        ax1[i].plot([chlor.min(), chlor.max()], [chlor.min(), chlor.max()], "k--", lw=2,
                   label='1:1 line')  # Identity line
        ax1[i].set_xlabel("Actual Chlorophyll Content")
        ax1[i].set_ylabel("Predicted Chlorophyll Content")
        ax1[i].set_title(f"Testing Set with {len(layer)} Layers")
        ax1[i].grid(True)
        ax1[i].legend()
        # Display metrics on plot
        train_metrics = f"MAE: {mae:.4f}\nR²: {r2:.4f}\nStd Res: {std_residuals:.4f}"
        ax1[i].text(0.05, 0.95, train_metrics, transform=ax1[i].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

        # residual plot per model
        ax2[i].scatter(y_test_pred, y_test - y_pred, color="green", alpha=0.6, label="Prediction Residuals")
        ax2[i].axhline(0, color="k", linestyle="--", lw=2, label='0 line')  # Horizontal line at y=0
        ax2[i].set_ylabel("Residuals")
        ax2[i].set_xlabel("Predicted Chlorophyll Content")
        ax2[i].set_title(f"Testing Set with {len(layer)} Layers")
        ax2[i].set_ylim(-20, 20)
        ax2[i].grid(True)
        ax2[i].legend()
        # Display metrics on Test Set plot
        test_metrics = f"MAE: {mae:.4f}\nR²: {r2:.4f}\nStd Res: {std_residuals:.4f}"
        ax2[i].text(0.05, 0.95, test_metrics, transform=ax2[i].transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    fig1.suptitle('Regression Plots for MLP Layers',fontweight='bold')
    fig2.suptitle('Residual Plots for MLP Layers',fontweight='bold')

    plt.tight_layout()
    plt.show()

    # plot the effect of accuracy vs number of hidden layers
    plt.plot(list(range(1,len(layerList)+1)), r2_train_scores,linestyle="-", marker=".")
    plt.axvline(list(range(1,len(layerList)+1))[np.argmax(r2_train_scores)], color="r", linestyle="--",label="Max Score")
    plt.title("Training Accuracy for Number of MLP Layers")
    plt.xlabel("Number of Layers")
    plt.ylabel("R²")
    plt.legend()
    plt.show()

    # -----------------Analyze Bias and Variance Tradeoff Between the 3 Models-----------------------

    # linear regression (ridge)
    y_pred_lr = ridge.predict(x_test) # predict on test data
    mae_lr, r2_lr, std_residuals_lr = compute_regression_metrics(y_test, y_pred_lr)

    # PLSR
    y_pred_plsr = pls.predict(x_test) # predict using test data
    mae_plsr, r2_plsr, std_residuals_plsr = compute_regression_metrics(y_test, y_pred_plsr) # testing metrics

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=layerList[2], activation='relu', max_iter=5000, random_state=42,
                       learning_rate_init=0.001) # initialize MLP regressor
    mlp.fit(x_train, y_train)  # train the MLP
    y_pred_mlp = mlp.predict(x_test) # predict on test set
    mae_mlp, r2_mlp, std_residuals_mlp = compute_regression_metrics(y_test, y_pred_mlp) # testing metrics

    # Define method names
    methods = [" Linear Regression (Ridge)", "PLSR", "MLP"]

    # Define MAE and Std of Residuals for each method
    mae_values = [mae_lr, mae_plsr, mae_mlp]
    std_residuals_values = [std_residuals_lr, std_residuals_plsr, std_residuals_mlp]

    # Define bar width
    bar_width = 0.4
    x = np.arange(len(methods))  # X locations for the groups

    # Create bar plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot bars for MAE and Std of Residuals
    bars1 = ax.bar(x - bar_width / 2, mae_values, bar_width, label="Bias (MAE)", color="blue", alpha=0.7)
    bars2 = ax.bar(x + bar_width / 2, std_residuals_values, bar_width, label=" Variance (Std Residuals)", color="orange", alpha=0.7)

    # Add labels and title
    ax.set_xlabel("Model Complexity")
    ax.set_ylabel("Error Value")
    ax.set_title("Bias and Variance vs Model Complexity")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()

    # Show the plot
    plt.show()




