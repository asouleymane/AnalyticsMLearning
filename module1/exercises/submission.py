import os, sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

DATASET_TEST = '/dsa/data/all_datasets/titanic_ML/titanic.test.csv'

def check(model):
    if not os.path.exists(DATASET_TEST):
        raise Exception('Test dataset not found. Please ask instructor for help.')
    
    if not isinstance(model, GaussianNB):
        raise TypeError('Expecting a GaussianNB model.')
        
    try:
        prediction = model.predict(pd.read_csv(DATASET_TEST))
    except Exception as e:
        raise RuntimeError('Unable to perform prediction on test dataset.', e)
        
    if prediction.shape != (419,):
        raise Exception('Resulting prediction has wrong dimension. Expecting: (419,) Received:', prediction.shape)

def snapshot(model):
    os.system('mkdir -p datasets')
    os.system('rm -rf datasets/*.npy')
    prediction = np.array(model.predict(pd.read_csv(DATASET_TEST)))
    np.save('datasets/submission.npy', prediction, allow_pickle=False, fix_imports=True)
    assert os.path.exists('datasets/submission.npy')
