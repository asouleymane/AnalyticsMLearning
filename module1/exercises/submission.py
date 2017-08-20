import os, sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

DATASET_TEST = '/dsa/data/all_datasets/titanic_ML/titanic.test.csv'
OUTPUT_PATH = './'

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
    os.system('mkdir -p %s' % OUTPUT_PATH)
    os.system('rm -rf %s' % os.path.join(OUTPUT_PATH, '*.npy'))
    prediction = np.array(model.predict(pd.read_csv(DATASET_TEST)))
    FNAME = os.path.join(OUTPUT_PATH, 'submission.npy')
    np.save(FNAME, prediction, allow_pickle=False, fix_imports=True)
    assert os.path.exists(FNAME)
