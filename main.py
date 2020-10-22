import numpy as np
from numpy import mean, std, sum, min, delete
from pandas import read_csv, DataFrame, concat

# import packages from sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

# imports from script files
from scripts.errors import compute_errors

# imports packages for plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
from time import time

# Hyper-parameters / CONSTANTS
N = 520 # N is the number of WAPS columns
SRC_NULL = 100 # This is the current null value 
DST_NULL = -98 # This is the new Null value
DATA_LOCATION = "data/TrainingData.csv" # Where the database is located 
MIN_WAPS = 9 # Minimum Wireless Access Points 
QUANTITATIVE_COLUMNS = ['LONGITUDE', 'LATITUDE'] # Labels for the Regression problem
CATEGORICAL_COLUMNS = ['FLOOR', 'BUILDINGID'] # Labels for the Classification problem

def data_loading_n_cleaning(datafile):

    data = read_csv(datafile)
    data.head()

    # 1. drop unnecessary columns 
    data.drop(columns=["SPACEID", "RELATIVEPOSITION", "USERID", "TIMESTAMP"], inplace=True) 

    # 2. drop corrupted data of phone id 
    data = data[data.PHONEID != 17]

    # 3. Splitting labels and WAPS 
    X = data.iloc[:, :N]
    Y = data.iloc[:, N:]

    # 4. Normalization:
    #   1. Changing the Original null value from 100 and any value lower than -98 to -98
    #   2. Normalizing WAPs values to be from 0 to 1, weak to strong signal respectively.
    # Change null value to new null value and set all lower values to it.
    X.replace(SRC_NULL, DST_NULL, inplace=True)
    X [X < DST_NULL] = DST_NULL

    # Normalization
    X /= min(X)  # Dividing every WAPS by the lowest WAP value (-98), to get a value from 0 to 1, 0 being the strongest and 1 being the weakest
    X = 1 - X    # Inverting the value, making 0 the weakest and 1 the strongest
    
    return X, Y

def filter_out_low_WAPS(data, labels, num_samples):
    '''
    Removes all features with the number of  below thresh
    
    Parameters : data           : (DataFrame) Training Dataset
                 labels         : (DataFrame) Test Dataset
                 thresh         : (float) the number used to threshold the variance
    
    Returns    : new_data       : (DataFrame) Training Dataset
                 new_labels     : (DataFrame) Test Dataset
    '''   
    drop_rows = list()
    for i, x in enumerate(data):
        count = sum(x != 0)
        if count < num_samples:
            drop_rows.append(i)
            
    new_data = delete(data, drop_rows, axis=0)
    new_labels = delete(labels, drop_rows, axis=0)
        
    return new_data, new_labels

def declaring_KNN_model(k, algorithm, leaf_size):

    clf = KNeighborsClassifier(n_neighbors=k, algorithm=algorithm, leaf_size=leaf_size, p=1)
    regr = KNeighborsRegressor(n_neighbors=k, algorithm=algorithm, leaf_size=leaf_size, p=1)

    return clf, regr

def threshold_variance(x_train, x_test, thresh):
    '''
    Removes all features with variance below thresh
    
    Parameters : x_train  : (DataFrame) Training Dataset
                 x_test   : (DataFrame) Test Dataset
                 thresh   : (float) the number used to threshold the variance
    
    Returns    : x_train  : (DataFrame) Training Dataset
                 x_test   : (DataFrame) Test Dataset
    '''   
    variance_thresh = VarianceThreshold(thresh)
    x_train = variance_thresh.fit_transform(x_train)
    x_test = variance_thresh.transform(x_test)
    
    return x_train, x_test



def create_subreport(errors, M, phone_id=None):
    '''
    This function takes the set of errors and formats their output into a
    string so that it can be reported to the console, saved to a text file, or
    both.
    
    Parameters: errors     : (tuple) contains the four types of errors
                M          : (int) number of row elements in set
                phone_id   : (int or None) None implies that its a total report
                
    Returns:    subreport  : (str)
    '''
    build_missclass, floor_missclass, coords_err, std_err, coor_pr_err = errors
    
    mean_c = mean(coords_err)
    std_c = std(coords_err)
    
    build_error = build_missclass / M * 100 # Percent Error
    floor_error = floor_missclass / M * 100 # Percent Error
    
    if phone_id is not None:
        str1 = "Phone ID: %d" % phone_id
    else:
        str1 = "Totals Output:"
    str2 = "Mean Coordinate Error: %.2f +/- %.2f meters" % (mean_c, std_c)
    str3 = "Standard Error: %.2f meters" % std_err
    str4 = "Building Percent Error: %.2f%%" % build_error
    str5 = "Floor Percent Error: %.2f%%" % floor_error
    
    if coor_pr_err != "N/A":
        str6 = "Prob that Coordinate Error Less than 10m: %.2f%%" %coor_pr_err    
    else:
        str6 = ""
    
    subreport = '\n'.join([str1, str2, str3, str4, str5, str6])
    
    return subreport

if __name__ == "__main__":

    tic = time() # Start program performance timer

    # Load the data subsets X: Data, Y: Labels 
    X, Y = data_loading_n_cleaning(DATA_LOCATION)
    
    # x_train_o : the training samples, y_train : the training labels,
    # x_test_o  : the testing samples,  y_test  : the testing labels 
    # It's important to note that the train_test_split() function takes input of types lists,
    # numpy arrays, scipy-sparse matrices or pandas dataframes, output type is NumPy array
    x_train_o, x_test_o, y_train, y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=0)

    # Filter out low Active WAPS 
    x_train_o, y_train = filter_out_low_WAPS(x_train_o, y_train, MIN_WAPS)
    x_test_o, y_test = filter_out_low_WAPS(x_test_o, y_test, MIN_WAPS)

    y_train = DataFrame(y_train, columns=Y.columns)
    y_test = DataFrame(y_test, columns=Y.columns)

    # Declaring KNN Classification and Regression model
    clf, regr = declaring_KNN_model(k = 1, algorithm = 'kd_tree', leaf_size = 50)

    # Variance Threshholding 
    x_train, x_test = threshold_variance(x_train_o, x_test_o, thresh=0.00001)
    
    # Fitting and making prediction for the Classification problem
    fit = clf.fit(x_train, y_train[CATEGORICAL_COLUMNS])
    prediction = fit.predict(x_test)

    # Converting prediction into a DataFrame
    clf_prediction = DataFrame(prediction, columns=CATEGORICAL_COLUMNS)

    # Fitting and making prediction for the Regression problem
    fit = regr.fit(x_train_o, y_train[QUANTITATIVE_COLUMNS])
    prediction = fit.predict(x_test_o)
    regr_prediction = DataFrame(prediction, columns=QUANTITATIVE_COLUMNS)

    # Prediction from both Classification and Regression 
    prediction = concat((clf_prediction, regr_prediction), axis=1)

    errors = compute_errors(prediction, y_test)

    print (errors[0])
    print (errors[1])
    print (errors[2][100:150])
    
    total_report = create_subreport(errors, y_test.shape[0])
    print('\n' + total_report + '\n')


    toc = time() # Report program performance timer
    print("Program Timer: %.2f seconds" % (toc-tic))