"""
Nate Peters
Hannah Smith
11/21/19
ISTA 331 HW7

This module classifies a dataset of 70,000 digitized handwritten digits with a 
SGD classifier and a Logistic Regression classifier.
"""

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def get_data():
    '''
    Gets data from dataset
    ---------------------------------------------
    PARAMETERS:
    N/A
    RETURNS:
        (array) - data and labels from dataset
    '''
    mnist = sio.loadmat('mnist-original.mat')
    return np.transpose(mnist['data']), mnist['label']

def get_train_and_test_sets(X, y):
    '''
    Gets training and testing sets from X and y variables
    ---------------------------------------------
    PARAMETERS:
        X(array) - features 
        y(array) - labels
    RETURNS:
        X_training(array) - training X
        X_testing(array) - testing X
        y_training(array) - training y
        y_testing(array) - testing y
    '''
    X_training_temp = X[:60000]
    y_training_temp = np.transpose(y[:60000])
    X_testing = X[60000:]
    y_testing =y[60000:]

    X_training = np.zeros_like(X_training_temp)
    y_training = np.zeros_like(y_training_temp)
    random_indices = list(np.random.permutation(60000))

    for i in range(len(random_indices)):
        X_training[i] = X_training_temp[random_indices[i]]
        y_training[i] = y_training_temp[random_indices[i]]

    return X_training, X_testing, y_training, y_testing

def train_to_data(X, y, model_name):
    '''
    Fits data to either SGDClassifier model or LogisticsRegression model
    ---------------------------------------------
    PARAMETERS:
        X(array) - training X        
        y(array) - training y
        model_name(str) - name of classifier model
    RETURNS:
        (obj) - classifier model 
    '''
    if model_name == 'SGD':
        clf = SGDClassifier(max_iter=50, tol=0.001)
    elif model_name == 'abc':
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    clf.fit(X,y)
    return clf

def get_confusion_matrix(model, X, y):
    '''
    Gets confusion matrix of data
    ---------------------------------------------
    PARAMETERS:
        model(obj) - classifier model
        X(array) - training X        
        y(array) - training y
    RETURNS:
        (array) - confusion matrix of actual y and predicted y 
    '''
    return confusion_matrix(y, cross_val_predict(model, X, y, cv=5)) #5 fold cross-validation predictor

def probability_matrix(conf_matrix):
    '''
    Gets probability matrix from confusion matrix
    ---------------------------------------------
    PARAMETERS:
    (array) - confusion matrix
    RETURNS:
        (array) - estimated condtional probabilities that j was predicted given i was label
    '''
    prob_matrix = np.zeros_like(conf_matrix, dtype=float)
    for i in range(len(prob_matrix)):
        for j in range(len(prob_matrix[i])):
            prob_matrix[i,j] = round(conf_matrix[i,j]/np.sum(conf_matrix[i]), 3)
    return prob_matrix

def plot_probability_matrices(sgd_pmat, soft_pmat):
    '''
    Plots two probability matrices, one from SGDClassifier and the other from
    Softmax Regression
    ---------------------------------------------
    PARAMETERS:
        sgd_pmat(array) - probability matrix using SGD classifier
        soft_pmat(array) - probability matrix using Logistic Regression classifier 
    RETURNS:
        N/A
    '''
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.matshow(sgd_pmat, cmap='Greys')
    ax1.set_title('SGDClassifier\n')
    ax2.matshow(soft_pmat, cmap='Greys')
    ax2.set_title('SoftmaxRegression\n')

def main():
    '''
    Gets features and labels from dataset, splits each up into 
    separate training and testing sets, fits training data to 
    classifier model(s), gets confusion matrices of data, gets probability
    matrices of confusion matrix, prints matrices, and shows the graphical
    depictions of the confusion matrices
    '''
    X, y = get_data()
    X_train, X_test, y_train, y_test = get_train_and_test_sets(X,y)
    sgd_model = train_to_data(X_train[:1000], y_train[:1000], 'SGD')
    soft_model = train_to_data(X_train[:1000], y_train[:1000], 'abc')

    sgd_cmat = get_confusion_matrix(sgd_model, X_train[:1000], y_train[:1000])
    soft_cmat = get_confusion_matrix(soft_model, X_train[:1000], y_train[:1000])
    sgd_pmat = probability_matrix(sgd_cmat)
    soft_pmat = probability_matrix(soft_cmat)
    
    print()
    print()
    for mod in (('SGDClassifier:', probability_matrix(sgd_cmat)),
                        ('Softmax:', probability_matrix(soft_cmat))):
        print(*mod, sep = '\n')
        print()

    plot_probability_matrices(sgd_pmat, soft_pmat)
    plt.show()

if __name__ == "__main__":
    main()