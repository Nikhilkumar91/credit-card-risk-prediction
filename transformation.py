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
logger =function('transformation')

def log_transform(X_train_num,X_test_num):
    try:
        for i in X_train_num.columns:
            X_train_num[i+'_log']=np.log(X_train_num[i]+1)
            X_test_num[i + '_log'] = np.log(X_test_num[i] +1)

        logger.info(f'Transformation Successfully completed : {X_train_num.columns}')
        logger.info(f'Tranformation Successfully Completed : {X_test_num.columns}')

        f=[]
        for j in X_train_num.columns:
            if '_log' not in j:
                f.append(j)
        X_train_num=X_train_num.drop(f,axis=1)
        X_test_num=X_test_num.drop(f,axis=1)

        logger.info(f'After tranformation : {X_train_num.columns}')
        logger.info(f'After Transformation : {X_test_num.columns}')

        return X_train_num,X_test_num

    except Exception as e:
        er_ty,er_msg,er_lin=sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')



