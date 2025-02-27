import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylabel
from numpy.ma.core import zeros_like
import xgboost as xgb
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight


import pca
import eda.eda as eda
import sklearn
from sklearn.metrics import roc_curve,auc


if __name__ == "__main__":

    # load the PaviaU class map
    mat = scipy.io.loadmat('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW3 - Supervised ML/PaviaU_gt.mat')
    classmap = mat['paviaU_gt']

    # load PaviaU hyperspectral data
    mat = scipy.io.loadmat('/Users/rjpearsall/Library/CloudStorage/GoogleDrive-rxp7504@g.rit.edu/My Drive/Imaging Science MS/Applied ML for Remote Sensing/Homework/HW3 - Supervised ML/PaviaU.mat')
    data = mat['paviaU']

    # ROSIS-3 center wavelengths
    wl = np.linspace(430,960,103)

    # list of class labels
    classes = list(range(1,10))

    # display RGB image of the data
    pca.display_rgb(data,wl)

    # pseudocolor of class map
    fig = plt.figure(figsize=(4,6))
    plt.imshow(classmap,cmap='Set1')
    plt.colorbar()
    plt.axis('off')
    plt.title('Class Map')
    plt.tight_layout()
    plt.show()

# -------------Exploratory Data Analysis--------------------

    # calculate band statistics
    stats = eda.calculate_band_statistics(data)

    # plot mean of each band
    plt.plot(wl,stats[:,0],label='Mean')
    plt.plot(wl,stats[:,1],label='Std')
    plt.title('Band Statistics')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Radiance [W/m²/sr]')
    plt.legend()
    plt.show()

    # make correlation matrix TODO: set k to full bands before posting
    k = 10
    cor = eda.correlation_matrix(data[:,:,0:k], wl[0:k],stats)

    # class map histogram
    # plt.hist(classmap.flatten(),bins=len(classes))
    # plt.xticks(np.arange(1, len(classes)))
    # plt.xlabel('Class')
    # plt.ylabel('Frequency')
    # plt.title('Class Map Histogram')
    # plt.show()

    # ---------- Binary Classification Using Logistic Regression --------------
    # reshape data
    data_reshaped = data.reshape(-1, data.shape[2])
    classmap_reshaped = classmap.flatten()

    # remove class 0
    data_reshaped = data_reshaped[classmap_reshaped>0,:]
    classmap_reshaped = classmap_reshaped[classmap_reshaped>0]

    # generate training and testing partitions
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data_reshaped,classmap_reshaped,test_size=0.3,train_size=0.7,random_state=42,shuffle=True,stratify=classmap_reshaped)

    # plot histograms of test and training to ensure they are balanced
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist(y_train,bins=np.arange(1,len(classes) + 2) - 0.5,edgecolor='black',align='mid')
    ax[0].set_xticks(np.arange(1,len(classes)+1))
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Class Map Histogram (Training Data)')
    ax[1].hist(y_test,bins=np.arange(1,len(classes) + 2) - 0.5,edgecolor='black',color='g')
    ax[1].set_xticks(np.arange(1,len(classes)+1))
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Class Map Histogram (Testing Data)')
    plt.tight_layout()
    plt.show()

    # bin data to two classes (1 = veg , 0 = not veg)
    y2_train = ((y_train == 2) | (y_train == 4)).astype(int)
    y2_test = ((y_test == 2) | (y_test == 4)).astype(int)

    # visualize 2 class balance
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist(y2_train,edgecolor='black')
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Class Map Histogram (Training Data)')
    ax[1].hist(y2_test,edgecolor='black',color='g')
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Class Map Histogram (Testing Data)')
    plt.tight_layout()
    plt.show()

    # grab band in B,G,R,R-edge,NIR
    sub_band_ideal = [450,550,650,750,850]
    x2_train = np.zeros([x_train.shape[0],len(sub_band_ideal)])
    x2_test = np.zeros([x_test.shape[0],len(sub_band_ideal)])
    j = 0
    for i in sub_band_ideal:
        x2_train[:,j] = x_train[:,np.argmin(np.abs(i-wl))]
        x2_test[:,j] = x_test[:,np.argmin(np.abs(i-wl))]
        j += 1

    # print sample size for training and test data for 2 class problem
    print('2 Class Training Sample Size: ',x2_train.shape[0])
    print('2 Class Test Sample Size: ',x2_test.shape[0])

    # logistic regression function

    # standardize input data
    x2_train = (x2_train - np.mean(x2_train,0)) / np.std(x2_train,0)
    x2_test = (x2_test - np.mean(x2_test,0)) / np.std(x2_test,0)

    # set learning rate
    lr = 0.1

    # set epochs
    epochs = 500

    # initialize loss
    loss = np.zeros([epochs,1])

    # initialize random parameters
    theta = np.random.random(x2_train.shape[1]) * 0.01

    for i in range(epochs):

        # sigmoid probability function
        y_prob = 1 / (1 + np.exp(-np.dot(x2_train,theta)))

        # update parameters
        theta = theta + (lr * np.mean((y2_train - y_prob).reshape(-1,1) * x2_train,0))

        # loss value
        loss[i] = np.mean((y2_train - y_prob)**2)

        # early stop if loss does not improve for 10 epochs
        if np.abs(loss[i] - loss[i-10])/loss[i] < 1e-5:
            print(f'Model converged after {i} epochs')
            break

    # plot loss
    plt.plot(loss)
    plt.title('Training Loss')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.xlim([0,i])
    plt.show()

    # TODO: Make this all a cleaner function
    #-------------- TESTING -----------
    # sigmoid probability function
    y_prob = 1 / (1 + np.exp(-np.dot(x2_test, theta)))

    # threshold predicted values
    y_predicted = (y_prob > 0.5).astype(int)

    print('--------------------BINARY CLASSIFICATION METRICS---------------------')

    # calculate mean accuracy
    mean_accuracy = np.mean(y_predicted == y2_test)
    print("Mean Accuracy: ",mean_accuracy)

    # mean per-class accuracy
    acc_0 = np.sum((y2_test == 0) & (y_predicted == 0)) / np.sum(y2_test == 0) # number of correct predictions divided by total in class 0
    acc_1 = np.sum((y2_test == 1) & (y_predicted == 1)) / np.sum(y2_test == 1) # number of correct predictions divided by total in class 1
    mpca = 0.5 * (acc_0 + acc_1) # mean of accuracy between both classes
    print("MPCA: ",mpca)

    # precision (how many of predicted positives were correct - about being correct when predicting positive)
    tp = np.sum((y2_test == 1) & (y_predicted == 1)) # true positive
    tn = np.sum((y2_test == 0) & (y_predicted == 0)) # true negative
    fp = np.sum((y2_test == 0) & (y_predicted == 1)) # false positive
    fn = np.sum((y2_test == 1) & (y_predicted == 0)) # false negative
    precision = tp / (tp + fp)
    print("Precision: ",precision)

    # recall (how many actual positives were correctly labeled - about catching all the positives)
    recall = tp / (tp + fn)
    print("Recall: ",recall)

    #F1 Score
    f1 = 2 / ( (1/precision) + (1/recall) )
    print("F1: ",f1)

    # ROC Curves TODO: clean this up
    # tpr = tp / (tp + fn) # true positive rate (recall)
    # fpr = fp / (fp + tn) # false positive rate
    y_true = y2_test

    # Define thresholds (sorted unique values)
    thresholds = np.linspace(0,1,100)

    # Initialize lists to store FPR and TPR values
    tpr_list = []
    fpr_list = []

    # Compute TPR and FPR for each threshold
    for t in thresholds:
        # Convert probabilities to binary predictions
        y_pred = (y_prob >= t).astype(int)

        # Compute TP, FP, TN, FN
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        # Compute TPR and FPR
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)

        # Store values
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    # Convert lists to numpy arrays
    tpr = np.array(tpr_list)
    fpr = np.array(fpr_list)

    fpr_sklearn, tpr_sklearn, _ = roc_curve(y2_test, y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="Manual ROC")
    plt.plot(fpr_sklearn, tpr_sklearn, label="sklearn ROC")
    # plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.grid()
    plt.show()

    # calculate AUC TODO: try this with trapezoid rule
    auc1 = 1/len(thresholds) * np.sum(tpr)
    print("AUC: ",auc1)
    auc2 = auc(fpr, tpr)
    print("AUC2: ",auc2)

    # ----------Boosting / Bagging Using XGBoost----------------------

    # grab band in B,G,R,R-edge,NIR
    xboost_train = np.zeros([x_train.shape[0],len(sub_band_ideal)])
    xboost_test = np.zeros([x_test.shape[0],len(sub_band_ideal)])
    feature_names = np.zeros(len(sub_band_ideal), dtype=object)
    j = 0
    for i in sub_band_ideal:
        xboost_train[:,j] = x_train[:,np.argmin(np.abs(i-wl))]
        xboost_test[:,j] = x_test[:,np.argmin(np.abs(i-wl))]
        feature_names[j] = f'{np.round(wl[np.argmin(np.abs(i-wl))])} nm'
        j += 1

    # standardize input data
    xboost_train = (xboost_train - np.mean(xboost_train,0)) / np.std(xboost_train,0)
    xboost_test = (xboost_test - np.mean(xboost_test,0)) / np.std(xboost_test,0)

    # relabel training data for xgboost
    y_train = y_train - np.min(y_train)
    y_test = y_test - np.min(y_test)

    # dmatrix format for xgboost
    dtrain = xgb.DMatrix(xboost_train, label=y_train)
    dtest = xgb.DMatrix(xboost_test, label=y_test)

    num_classes = 9  # Number of classes

    params = {
        "objective": "multi:softmax",  # Multi-class classification
        "num_class": num_classes,
        "eval_metric": "mlogloss",  # Multi-class log loss
        "eta": 0.1,  # Learning rate
        "max_depth": 6,  # Depth of trees
        "subsample": 0.8,  # Subsample ratio
        "colsample_bytree": 0.8,  # Feature fraction per tree
        "seed": 42  # Reproducibility
    }

    num_rounds = 100  # Number of boosting rounds

    # train the model
    model = xgb.train(params, dtrain, num_rounds)

    # make predictions on test data
    y_pred = model.predict(dtest).astype(int)  # Returns class labels

    # Plot feature importance
    ax = xgb.plot_importance(model, importance_type="gain")
    fig = plt.gcf()  # Get current figure
    fig.set_size_inches(9, 5)  # Set the figure size
    plt.yticks(ticks=range(len(feature_names)),labels=feature_names, rotation=45)
    plt.title("Feature Importance")
    plt.show()

    # scale classes back to original values
    y_test = y_test + 1
    y_pred = y_pred + 1

    # plot confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(1,num_classes+1):
        for j in range(1,num_classes+1):
            confusion_matrix[i-1,j-1] = np.sum((y_test == i) & (y_pred == j))

    # normalize confusion matrix
    confusion_matrix = confusion_matrix / np.where(confusion_matrix.sum(axis=1)[:, np.newaxis] == 0, 1,confusion_matrix.sum(axis=1)[:, np.newaxis])
    plt.imshow(confusion_matrix,cmap="Blues")
    plt.colorbar()
    plt.title("XGBoost Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(range(num_classes),labels=range(1,num_classes+1))
    plt.yticks(range(num_classes),labels=range(1,num_classes+1))
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{confusion_matrix[i, j]:.2f}",
                     ha="center", va="center", color="black" if confusion_matrix[i, j] < 0.5 else "white")

    plt.show()

    print('--------------------UNBALANCED MULTICLASS METRICS---------------------')

    # calculate mean accuracy
    mean_accuracy = np.mean(y_pred == y_test)
    print("Mean Accuracy: ",mean_accuracy)

    # mean per-class accuracy
    acc = np.zeros(num_classes)
    for i in range(1,num_classes+1):
        acc[i-1] = np.sum((y_test == i) & (y_pred == i)) / np.sum(y_test == i) # number of correct predictions divided by total in class
    mpca = np.mean(acc) # mean accuracy of all classes
    print("MPCA: ",mpca)

    # precision (how many of predicted positives were correct - about being correct when predicting positive)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    for i in range(1,num_classes+1):
        tp = np.sum((y_test == i) & (y_pred == i)) # true positive
        tn = np.sum((y_test != i) & (y_pred != i)) # true negative
        fp = np.sum((y_test != i) & (y_pred == i)) # false positive
        fn = np.sum((y_test == i) & (y_pred != i)) # false negative
        precision[i-1] = tp / (tp + fp)
        recall[i-1] = tp / (tp + fn) # recall (how many actual positives were correctly labeled - about catching all the positives)

    print("Precision: ",precision)
    print("Recall: ",recall)

    # F1 Score
    f1 = 2 / ((1 / precision) + (1 / recall))
    print("F1: ",f1)

    # ---------Balance the Dataset Using Under and Oversampling---------
    class_counts = np.bincount(classmap_reshaped) # how many pixels in each class
    class_counts = class_counts[1:] # ignore class 0
    class_count_avg = int(np.mean(class_counts)) # average of all bin counts

    # initialize as lists (using all original data as input for balancing)
    X_balanced = []
    Y_balanced = []
    for i in range(num_classes):
        if class_counts[i] > class_count_avg: # if bins are higher than the average
            # randomly sample pixels of each class to match average count value
            x_resampled, y_resampled = resample(data_reshaped[classmap_reshaped == i+1,:], classmap_reshaped[classmap_reshaped == i+1], n_samples=class_count_avg, random_state=42)
        else: # if bins are less than average
            # randomly sample minority classes with replacement to provide duplicates
            x_resampled, y_resampled = resample(data_reshaped[classmap_reshaped == i+1,:], classmap_reshaped[classmap_reshaped == i+1], n_samples=class_count_avg, random_state=42,replace=True)

        X_balanced.append(x_resampled)
        Y_balanced.append(y_resampled)

    # Convert to numpy arrays
    X_balanced = np.vstack(X_balanced)
    Y_balanced = np.hstack(Y_balanced)

    # # Plot class distribution
    # plt.hist(Y_balanced, bins=np.arange(1, num_classes + 2) - 0.5, edgecolor='black', align='mid')
    # plt.xticks(np.arange(1, num_classes + 1))
    # plt.xlabel("Class Label")
    # plt.ylabel("Frequency")
    # plt.title("Balanced Class Distribution")
    # plt.show()

    # split balanced data into training and testing partitions
    x_balanced_train, x_balanced_test, y_balanced_train, y_balanced_test = sklearn.model_selection.train_test_split(X_balanced,Y_balanced,test_size=0.3,train_size=0.7,random_state=42,shuffle=True,stratify=Y_balanced)

    # plot histograms of test and training to ensure they are balanced
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].hist(y_balanced_train,bins=np.arange(1,len(classes) + 2) - 0.5,edgecolor='black',align='mid')
    ax[0].set_xticks(np.arange(1,len(classes)+1))
    ax[0].set_xlabel('Class')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Balanced Class Map Histogram (Training Data)')
    ax[1].hist(y_balanced_test,bins=np.arange(1,len(classes) + 2) - 0.5,edgecolor='black',color='g')
    ax[1].set_xticks(np.arange(1,len(classes)+1))
    ax[1].set_xlabel('Class')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Balanced Class Map Histogram (Testing Data)')
    plt.tight_layout()
    plt.show()


    # ------train a new model using balanced data-------

    # grab band in B,G,R,R-edge,NIR
    xboost_train = np.zeros([x_balanced_train.shape[0],len(sub_band_ideal)])
    xboost_test = np.zeros([x_balanced_test.shape[0],len(sub_band_ideal)])
    j = 0
    for i in sub_band_ideal:
        xboost_train[:,j] = x_balanced_train[:,np.argmin(np.abs(i-wl))]
        xboost_test[:,j] = x_balanced_test[:,np.argmin(np.abs(i-wl))]
        j += 1

    # standardize input data
    xboost_train = (xboost_train - np.mean(xboost_train,0)) / np.std(xboost_train,0)
    xboost_test = (xboost_test - np.mean(xboost_test,0)) / np.std(xboost_test,0)

    # relabel training data for xgboost (start at 0)
    y_balanced_train = y_balanced_train - np.min(y_balanced_train)
    y_balanced_test = y_balanced_test - np.min(y_balanced_test)

    # dmatrix format for xgboost
    dtrain = xgb.DMatrix(xboost_train, label=y_balanced_train)
    dtest = xgb.DMatrix(xboost_test, label=y_balanced_test)

    # train the model
    model = xgb.train(params, dtrain, num_rounds)

    # make predictions on test data
    y_pred = model.predict(dtest).astype(int)  # Returns class labels

    # scale classes back to original values
    y_balanced_test = y_balanced_test + 1
    y_pred = y_pred + 1

    # plot confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(1,num_classes+1):
        for j in range(1,num_classes+1):
            confusion_matrix[i-1,j-1] = np.sum((y_balanced_test == i) & (y_pred == j))

    # normalize confusion matrix
    confusion_matrix = confusion_matrix / np.where(confusion_matrix.sum(axis=1)[:, np.newaxis] == 0, 1,confusion_matrix.sum(axis=1)[:, np.newaxis])
    plt.imshow(confusion_matrix,cmap="Blues")
    plt.colorbar()
    plt.title("XGBoost Confusion Matrix Using Balanced Data (Normalized)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(range(num_classes),labels=range(1,num_classes+1))
    plt.yticks(range(num_classes),labels=range(1,num_classes+1))
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{confusion_matrix[i, j]:.2f}",
                     ha="center", va="center", color="black" if confusion_matrix[i, j] < 0.5 else "white")

    plt.show()

    print('--------------------BALANCED MULTICLASS METRICS---------------------')
    # calculate mean accuracy
    mean_accuracy = np.mean(y_pred == y_balanced_test)
    print("Mean Accuracy: ",mean_accuracy)

    # mean per-class accuracy
    acc = np.zeros(num_classes)
    for i in range(1,num_classes+1):
        acc[i-1] = np.sum((y_balanced_test == i) & (y_pred == i)) / np.sum(y_balanced_test == i) # number of correct predictions divided by total in class
    mpca = np.mean(acc) # mean accuracy of all classes
    print("MPCA: ",mpca)

    # precision (how many of predicted positives were correct - about being correct when predicting positive)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    for i in range(1,num_classes+1):
        tp = np.sum((y_balanced_test == i) & (y_pred == i)) # true positive
        tn = np.sum((y_balanced_test != i) & (y_pred != i)) # true negative
        fp = np.sum((y_balanced_test != i) & (y_pred == i)) # false positive
        fn = np.sum((y_balanced_test == i) & (y_pred != i)) # false negative
        precision[i-1] = tp / (tp + fp)
        recall[i-1] = tp / (tp + fn) # recall (how many actual positives were correctly labeled - about catching all the positives)

    print("Precision: ",precision)
    print("Recall: ",recall)

    # F1 Score
    f1 = 2 / ((1 / precision) + (1 / recall))
    print("F1: ",f1)






