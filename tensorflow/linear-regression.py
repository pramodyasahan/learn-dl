from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

dftrain = pd.read_csv('train.csv')
dfeval = pd.read_csv('eval.csv')
y_train = dftrain['survived'].values
y_eval = dftrain['survived'].values
print(y_eval)
