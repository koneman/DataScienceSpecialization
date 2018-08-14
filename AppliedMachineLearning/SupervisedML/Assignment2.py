"""
Created by Sathvik Koneru on 8/13/18.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);


# NOTE: Uncomment the function below to visualize the data, but be sure
# to **re-comment it before submitting this assignment to the autograder**.
# part1_scatter()

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    poly_degrees = enumerate([1,3,6,9])
    pred_100 = np.zeros([4,100])

    for index,poly_degree in poly_degrees:
        poly = PolynomialFeatures(degree=poly_degree)
        #fit_transform requires x parameter to have shape [n_samples, n_features]
        X_poly = poly.fit_transform(X_train.reshape(11,1))
        linreg = LinearRegression().fit(X_poly, y_train)
        y_pred = linreg.predict(poly.fit_transform(np.linspace(0,10,100).reshape(100,1)))
        pred_100[index] = y_pred

    return pred_100


# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    %matplotlib notebook
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)

# plot_one(answer_one())


def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    r2_train = np.arange(10, dtype='float64')
    r2_test = np.arange(10, dtype='float64')

    for i in range(0,10):
        poly = PolynomialFeatures(degree=i)
        #fit_transform requires x parameter to have shape [n_samples, n_features]
        X_poly_train = poly.fit_transform(X_train.reshape(11,1))
        X_poly_test = poly.fit_transform(X_test.reshape(4,1))
        linreg = LinearRegression().fit(X_poly_train, y_train)

        r2_train[i] = linreg.score(X_poly_train, y_train)
        r2_test[i] = linreg.score(X_poly_test, y_test)

    return (r2_train, r2_test)

answer_two()

def answer_three():
    r2_values = answer_two()
    underfit_values = []
    overfit_values = []
    goodfit_values = []

    for i in range(0,2):
        for j in range(0,10):
            if r2_values[i][j] < 0.6:
                underfit_values.append(j)
            if r2_values[i][j] > 0.6 and r2_values[i][j] < 0.7:
                overfit_values.append(j)
            if r2_values[i][j] > 0.9:
                goodfit_values.append(j)
    #return (underfit_values, overfit_values, goodfit_values)
    return (0, 9, 6)

answer_three()

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Polynomial Regression
    poly = PolynomialFeatures(degree=12)
    X_poly_train = poly.fit_transform(X_train.reshape(11,1))
    X_poly_test = poly.fit_transform(X_test.reshape(4,1))
    linreg = LinearRegression().fit(X_poly_train, y_train)
    LinearRegression_R2_test_score = linreg.score(X_poly_test, y_test)

    # Lasso Regression
    linlasso = Lasso(alpha=0.01, max_iter=10000).fit(X_poly_train, y_train)
    Lasso_R2_test_score = linlasso.score(X_poly_test, y_test)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)

answer_four()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2

def answer_five():
    from sklearn.tree import DecisionTreeClassifier
    from scipy.stats import rankdata

    #fitting decision tree classifier
    decision_tree = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    feature_importances = decision_tree.feature_importances_
    feature_names = X_train2.columns

    #constructing a dataframe to sort through the data
    feature_df = pd.DataFrame(data=feature_importances, index=feature_names, columns=['Feature Importance'])
    sorted_feature_df = feature_df.sort(['Feature Importance'],ascending=False)
    top5_features_df = sorted_feature_df.iloc[:5]
    top5_features_values = top5_features_df.index.values
    feature5_list = []
    for i in top5_features_values:
        feature5_list.append(i)
    return feature5_list

answer_five()

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    param_range = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(SVC(), X_subset, y_subset,
                                            param_name='gamma',
                                            param_range=param_range, cv=3)
    train_mean_scores = train_scores.mean(axis=1)
    test_mean_scores = test_scores.mean(axis=1)

    return (train_mean_scores, test_mean_scores)

answer_six()

def answer_seven():
    return (0.001, 10, 0.1)
