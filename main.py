import os
import numpy as np
import glob
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

MAIN_DIRECTORY = os.getcwd()

def load_data():
    """Function to load the training and test data"""
    stack_empty = 0
    path =  MAIN_DIRECTORY + '/training_data/'
    # Training Data folders
    folders = ['Session01', 'Session05', 'Session06', 'Session07', 'Session12']
    length = len(folders)
    for folder in folders:
        print('Loading data from ' + folder)
        new_directory = path + folder
        os.chdir(new_directory)
        timestamp = np.loadtxt(open('time.txt', 'r'), delimiter=',')
        features = np.loadtxt(open('features.csv', 'r'), delimiter=',')
        detections = np.int32(np.loadtxt(open('detection.txt', 'r'), delimiter=','))
        if stack_empty == 0:
            x_train = features
            time_stack = timestamp
            y_train = detections
            stack_empty = 1
        else:
            x_train = np.vstack((x_train, features))
            time_stack = np.hstack((time_stack, timestamp))
            y_train = np.hstack((y_train, detections))

    # Load the testing data , using one of the training Sessions as testing for now
    new_directory = path + 'Session13'
    os.chdir(new_directory)
    x_test = np.loadtxt(open('features.csv', 'r'), delimiter=',')
    y_test = np.int32(np.loadtxt(open('detection.txt', 'r'), delimiter=','))
    return x_train, y_train, x_test, y_test

def remove_nan(x):
    """Function to replace missing values from dataset"""
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    # Transform training data
    imputer = imputer.fit(x)
    x = imputer.transform(x)
    return x


def data_preprocessing(x_train, x_test):
    """Function to standardize training and testing data"""
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    return x_train, x_test


def main():
    # Enable if you want to remove NaNs from and perform standardization over training and test data
    preprocessing = 1
    # Enable if you want to use stratified folds for cross-validation
    stratified_fold = 1
    
    # Load Data
    x_train, y_train, x_test, y_test = load_data()

    if preprocessing:
        x_train = remove_nan(x_train)
        x_test = remove_nan(x_test)
        x_train, x_test = data_preprocessing(x_train, x_test)

    # Parameters over which hyperparameter will be performed
    parameters = {'alpha': (0.1, 0.01, 0.001, 0.0001),
                  'loss': ('log', 'hinge'),
                  'tol': (1e-3, 1e-4, 1e-5)
                  }
    # Define the model
    model = SGDClassifier(penalty='l2', max_iter=1000, learning_rate='optimal',
                          eta0=0.001, shuffle=True)
    if stratified_fold:
        cv = StratifiedKFold(n_splits=5)
    else:
        cv = 5
    # Hyperparameter Tuning
    clf = GridSearchCV(model, parameters, verbose=1, cv=cv)
    # Save the model
    pickle.dump(clf,open("brb_detector_clf.pkl","wb"))
    
    # Testing Scores
    score = clf.score(x_test, y_test)
    print(clf.best_estimator_)
    print("Score: " + str(score))


if __name__ == '__main__':
    main()
