import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
database = datasets.load_digits()
print(len(database.images))
