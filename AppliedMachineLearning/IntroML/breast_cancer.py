"""
Created by Sathvik Koneru on 8/1/18.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# reference to breast cancer dataset
cancer = load_breast_cancer()


# Convert the sklearn.dataset cancer to a DataFrame.
# This function should return a (569, 31)
def answer_one():
    cancer_df = pd.DataFrame(data=cancer.data, columns=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
                                                        'mean smoothness', 'mean compactness', 'mean concavity',
                                                        'mean concave points', 'mean symmetry',
                                                        'mean fractal dimension',
                                                        'radius error', 'texture error', 'perimeter error',
                                                        'area error',
                                                        'smoothness error', 'compactness error', 'concavity error',
                                                        'concave points error', 'symmetry error',
                                                        'fractal dimension error',
                                                        'worst radius', 'worst texture', 'worst perimeter',
                                                        'worst area',
                                                        'worst smoothness', 'worst compactness', 'worst concavity',
                                                        'worst concave points', 'worst symmetry',
                                                        'worst fractal dimension'])
    cancer_df['target'] = cancer.target
    return cancer_df


# What is the class distribution? (i.e. how many instances of malignant (encoded 0) and how many benign (encoded 1)?)
# This function should return a Series named target of length 2 with integer values and index = ['malignant', 'benign']
def answer_two():
    # create a cancer dataframe from answer one
    cancer_df = answer_one()
    malignant_counter = 0
    benign_counter = 0
    for target_value in cancer_df['target']:
        if target_value == 0:
            malignant_counter += 1
        elif target_value == 1:
            benign_counter += 1

    target = pd.Series(data=[malignant_counter, benign_counter], index=['malignant', 'benign'])
    return target


# Split the DataFrame into X (the data) and y (the labels).
# This function should return a tuple of length 2: (X, y), where
# X, a pandas DataFrame, has shape (569, 30) and y, a pandas Series, has shape (569,).
def answer_three():
    cancer_df = answer_one()
    X = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
    y = pd.Series(data=cancer.target)
    return X, y


# Using train_test_split, split X and y into training and test sets (X_train, X_test, y_train, and y_test).
from sklearn.model_selection import train_test_split
def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return X_train, X_test, y_train, y_test


# Using KNeighborsClassifier, fit a k-nearest neighbors (knn)
# classifier with X_train, y_train and using one nearest neighbor (n_neighbors = 1).
from sklearn.neighbors import KNeighborsClassifier
def answer_five():
    X_train, X_test, y_train, y_test = answer_four()
    knn = KNeighborsClassifier(n_neighbors=1)
    fit = knn.fit(X_train, y_train)
    return fit


# Using your knn classifier, predict the class label using the mean value for each feature
def answer_six():
    cancerdf = answer_one()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn = answer_five()
    mean_prediction = knn.predict(means)
    return mean_prediction


# Using your knn classifier, predict the class labels for the test set X_test.
def answer_seven():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    test_prediction = knn.predict(X_test)
    return test_prediction


# Find the score (mean accuracy) of your knn classifier using X_test and y_test.
def answer_eight():
    X_train, X_test, y_train, y_test = answer_four()
    knn = answer_five()
    test_score = knn.score(X_test, y_test)
    return test_score


def accuracy_plot():
    import matplotlib.pyplot as plt

    %matplotlib notebook

    X_train, X_test, y_train, y_test = answer_four()

    # Find the training and testing accuracies by target value (i.e. malignant, benign)
    mal_train_X = X_train[y_train==0]
    mal_train_y = y_train[y_train==0]
    ben_train_X = X_train[y_train==1]
    ben_train_y = y_train[y_train==1]

    mal_test_X = X_test[y_test==0]
    mal_test_y = y_test[y_test==0]
    ben_test_X = X_test[y_test==1]
    ben_test_y = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_train_X, mal_train_y), knn.score(ben_train_X, ben_train_y),
              knn.score(mal_test_X, mal_test_y), knn.score(ben_test_X, ben_test_y)]


    plt.figure()

    # Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0','#4c72b0','#55a868','#55a868'])

    # directly label the score onto the bars
    for bar in bars:
        height = bar.get_height()
        plt.gca().text(bar.get_x() + bar.get_width()/2, height*.90, '{0:.{1}f}'.format(height, 2),
                     ha='center', color='w', fontsize=11)

    # remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    # remove the frame of the chart
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.xticks([0,1,2,3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)



accuracy_plot()

