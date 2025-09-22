import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.model_selection import train_test_split
import sys
import warnings
warnings.filterwarnings('ignore')
#from random_sample import random_values
from loggin import function
from scipy.stats import pearsonr
logger =function('hypothesis')

def hypothesis_testing(train_num,test_num,train_dep,test_dep):
    try:
        logger.info(f'{test_dep.shape}')
        logger.info(f'{train_dep.shape}')
        logger.info(f'Dependent variable Unique labels are : {train_dep.unique()} :')
        train_dep = train_dep.map({'Good': 1, 'Bad': 0}).astype(int)
        logger.info(f'{test_dep.isnull().sum()}')
        test_dep= test_dep.map({'Good': 1, 'Bad': 0}).astype(int)
        logger.info(f'Dependent variable Unique labels are : {train_dep.unique()} ---->{train_dep.sample(5)}----->{train_num.columns}')
        c=[]
        for i in train_num.columns:
            co=pearsonr(train_num[i],train_dep)
            c.append(co)
        c = np.array(c)
        logger.info(f'Hypothesis Testing : {c}')
        result = pd.Series(c[:, 1], index=train_num.columns)

        #To get which column has p value <0.05 we are represent using bar plot
        '''result.sort_values(ascending=True).plot.bar()
        plt.show()'''
        train_num = train_num.drop(['DebtRatio_log_trim'], axis=1)
        test_num = test_num.drop(['DebtRatio_log_trim'], axis=1)
        logger.info(f'Train Column Names : {train_num.columns}')
        logger.info(f'Test Column Names : {test_num.columns}')
        return train_num, test_num
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

