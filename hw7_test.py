from hw7 import *
import hw7
import unittest

from contextlib import redirect_stdout
import io

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np, random
import matplotlib.pyplot as plt
import scipy.io as sio

"""
GRADING!!!!  Discuss rubric with SL's about plot each semester.
The test is for 80 pts.  The other 20 is the plot.
"""

"""
Files required:
hw7.py, mnist-original.mat
X_correct.npy, y_correct.npy
X_train_correct.npy, y_train_correct.npy
X_test_correct.npy, y_test_correct.npy
confusion_matrix_SGD_correct.npy, probability_matrix_SGD_correct.npy
main_out_correct.txt
"""

class TestAssignment7(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
    
    def test_get_data(self):
        X_correct = np.load('X_correct.npy')
        y_correct = np.load('y_correct.npy')
        X, y = get_data()
        self.assertTrue((X_correct == X).all())
        self.assertTrue((y_correct == y).all())
        
    def test_get_train_and_test_sets(self):   
        X = np.load('X_correct.npy')
        y = np.load('y_correct.npy')
        X_train_correct = np.load('X_train_correct.npy')
        X_test_correct = np.load('X_test_correct.npy')
        y_train_correct = np.load('y_train_correct.npy')
        y_test_correct = np.load('y_test_correct.npy')
        X_train, X_test, y_train, y_test = get_train_and_test_sets(X, y)
        self.assertTrue((X_train_correct == X_train).all())
        self.assertTrue((X_test_correct == X_test).all())
        self.assertTrue((y_train_correct == y_train).all())
        self.assertTrue((y_test_correct == y_test).all())
    
    def test_train_to_data(self):
        X = np.load('X_train_correct.npy')
        y = np.load('y_train_correct.npy')
        model = train_to_data(X[:100], y[:100], 'SGD')
        self.assertEqual(type(model), SGDClassifier)
        self.assertEqual(0.0001, model.alpha)
        self.assertEqual(0.001, model.tol)
        self.assertEqual(50, model.max_iter)
        self.assertEqual('hinge', model.loss)

        model = train_to_data(X[:100], y[:100], 'abc')
        self.assertEqual(type(model), LogisticRegression)
        self.assertEqual(1.0, model.C)
        self.assertEqual(0.0001, model.tol)
        self.assertEqual(100, model.max_iter)
        self.assertEqual('multinomial', model.multi_class)
        self.assertEqual('lbfgs', model.solver)
        
    def test_get_confusion_matrix(self):
        X = np.load('X_train_correct.npy')
        y = np.load('y_train_correct.npy')
        model = train_to_data(X[:1000], y[:1000], 'SGD')
        # really shouldn't be calling this on training data, but oh well
        np.random.seed(25)
        random.seed(25)
        cm = get_confusion_matrix(model, X[:1000], y[:1000])
        #np.save('confusion_matrix_SGD_correct.npy', cm)
        cm_correct = np.load('confusion_matrix_SGD_correct.npy')
        # self.assertTrue((cm_correct == cm).all())
        
    def test_probability_matrix(self):
        cm = np.load('confusion_matrix_SGD_correct.npy')
        pm = probability_matrix(cm)
        #np.save('probability_matrix_SGD_correct.npy', pm)
        pm_correct = np.load('probability_matrix_SGD_correct.npy')
        
        # make sure the confusion matrix hasn't been altered:
        self.assertTrue((np.load('confusion_matrix_SGD_correct.npy') == cm).all())
        self.assertTrue((pm_correct == pm).all())
         
test = unittest.defaultTestLoader.loadTestsFromTestCase(TestAssignment7)
results = unittest.TextTestRunner().run(test)
print('Correctness score = ', str((results.testsRun - len(results.errors) - len(results.failures)) / results.testsRun * 80) + ' / 80')
main()
























