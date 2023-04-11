#Standard libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM

from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline


columns = ['user','activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
read = pd.read_csv('WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header = None, names = columns)
read.head()
read.tail()
read.info()

read = read.dropna()
read.shape
