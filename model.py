"""
Resampling, Model Selection and Regularization with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import pandas as pd
from sklearn.datasets import load_diabetes

def load_data():
    return load_diabetes(as_frame=True, return_X_y=True)

def describe_data(X, y):
    return {
        'n': X.shape[0], 
        'p': X.shape[1], 
        'features': list(X.columns), 
        'y_mean': round(float(y.mean()), 2)
    }

# Step 2 - train_test
from sklearn.model_selection import train_test_split

def train_test(X, y, test_size=0.25, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Step 3 - validation_set_curve
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def poly_model(degree):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression())

def validation_set_curve(X, y, feature, degrees, random_state):
    X_train, X_val, y_train, y_val = train_test_split(X[[feature]], y, test_size=0.5, random_state=random_state)
    return {
        degree: round(
            mean_squared_error(y_val, poly_model(degree).fit(X_train, y_train).predict(X_val)), 
            1
            ) for degree in degrees
    }

def curve_spread(X, y, feature, degrees, seeds):
    # per degree, max - min of the validation MSE across seeds, rounded to 1 decimal
    min_mse = {degree : float('inf') for degree in degrees}
    max_mse = {degree : -float('inf') for degree in degrees}

    for seed in seeds:
        curve = validation_set_curve(X, y, feature, degrees, seed)
        for degree in degrees:
            if curve[degree] < min_mse[degree]:
                min_mse[degree] = curve[degree]
            if curve[degree] > max_mse[degree]:
                max_mse[degree] = curve[degree]

    return {
        degree: max_mse[degree] - min_mse[degree] for degree in degrees
    }

# Step 4 - cv_mse
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score

def cv_mse(estimator, X, y, k=5, random_state=0):
    # shuffled KFold; cross_val_score with neg MSE; return (mean, se) rounded to 2 decimals
    
    scores = -1 * cross_val_score(estimator, X, y, scoring='neg_mean_squared_error', cv=KFold(n_splits=k, shuffle=True, random_state=random_state))
    return np.mean(scores), np.std(scores, ddof=1) / np.sqrt(k)

def loocv_mse(estimator, X, y):
    # LeaveOneOut; mean squared error over held-out points, rounded to 2 decimals
    loo = LeaveOneOut()

    means = []

    for i, (train_index, test_index) in enumerate(loo.split(X)):
        mean, _ = cv_mse(estimator, X.iloc[train_index], y.iloc[train_index])
        means.append(mean)

    return np.round(np.mean(np.array(means)), 2)

# Step 5 - cv_spread_by_k
import numpy as np

def cv_spread_by_k(estimator, X, y, ks, seeds):
    # TODO: {k: (mean of cv estimate over seeds, std ddof=1 over seeds)} rounded to 1 decimal
    result = {}
    for k in ks:
        m = []
        for seed in seeds:
            m.append(cv_mse(estimator, X, y, k, seed)[0])
        result[k] = (np.mean(m), np.std(m, ddof=1))
    return result

def compare_with_loocv(estimator, X, y, ks, seeds):
    return {
        'loocv': loocv_mse(estimator, X, y),
         'kfold': cv_spread_by_k(estimator, X, y, ks, seeds)
    }

# Step 6 - bootstrap_coefficients (not yet solved)
# TODO: implement

# Step 7 - stepwise_path (not yet solved)
# TODO: implement

# Step 8 - score_path (not yet solved)
# TODO: implement

# Step 9 - one_se_rule (not yet solved)
# TODO: implement

# Step 10 - ridge_path (not yet solved)
# TODO: implement

# Step 11 - cv_curve (not yet solved)
# TODO: implement

# Step 12 - lasso_path (not yet solved)
# TODO: implement

# Step 13 - pcr_model (not yet solved)
# TODO: implement

# Step 14 - pls_model (not yet solved)
# TODO: implement

# Step 15 - fit_all (not yet solved)
# TODO: implement

# Step 16 - test_report (not yet solved)
# TODO: implement

