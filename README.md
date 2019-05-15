# Body-Rocking-Behaviour-Detection-ML

A project to identify and train a classical Machine Learning method for Body Rocking Behaviour Detection using inertial measurements from a wearable system. Two sensor modules, each consisting of acceleormeter and gyroscopes,worn on arm and wrist were used to record motion. The data is recorded from sessions of 1-2 hours long with annotations of when this behaviours was observed.

### Prerequisites

1. python 3.6
2. numpy
3. scikit-learn
4. Matlab R2018b

### Installing

You can pip install all the dependencies for the project.

## Running the model

1. Download the dataset [here]()
2. Run extractFeatures_stride.m with updated stride and window size (by default uses stride of 50 and window size of 200) in Matlab to perform feature extraction from training data. Features.csv should now be generated in each Session folder. 
3. Run main.py to perform hyperparameter tuning to find the best model

## Authors

* **Bharath C Renjith** -[cr-bharath](https://github.com/cr-bharath)
