import os, sys
import itertools
import random
import numpy as np
import pandas as pd

dataset = pd.DataFrame(np.random.rand(8, 5), columns = ['float', 'int', 'yes/no', 'date', 'categorical'])
dataset['int'] = list(map(lambda x: int(x*10), dataset['int']))
dataset['yes/no'] = list(map(lambda x: 'Yes' if x > 0.5 else 'No', dataset['yes/no']))
from datetime import datetime, timedelta
dataset['date'] = list(map(lambda x: (datetime.today() + timedelta(days=int(x))).strftime("%Y-%m-%d"), dataset['int']))
categories = 'ABCDEFG'
dataset['categorical'] = list(map(lambda x: categories[x%len(categories)], dataset['int']))
dataset.to_csv('processing_examples.csv')