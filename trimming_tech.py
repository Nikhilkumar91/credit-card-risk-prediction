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
logger =function('trimming_tech')

def trimming_technique(X_train_num,X_test_num):
    try:
        for i in X_train_num.columns:
            iqr=X_test_num[i].quantile(0.75)-X_test_num[i].quantile(0.25)
            upper_limit=X_test_num[i].quantile(0.75)+(1.5*iqr)
            lower_limit = X_test_num[i].quantile(0.25) - (1.5*iqr)
            X_train_num[i+'_trim']=np.where(X_train_num[i]>upper_limit,upper_limit,np.where(X_train_num[i]<lower_limit,lower_limit,X_train_num[i]))
            X_test_num[i + '_trim'] = np.where(X_test_num[i] > upper_limit, upper_limit,
                                                np.where(X_test_num[i] < lower_limit, lower_limit, X_test_num[i]))

        logger.info(f'After Successfull trimming : {X_train_num.columns} ')
        logger.info(f'After Succeessful trimming : {X_test_num.columns}')

        f=[]
        for j in X_train_num:
            if '_trim' not in j:
                f.append(j)

        X_train_num=X_train_num.drop(f,axis=1)
        X_test_num=X_test_num.drop(f,axis=1)

        logger.info(f'After Successfull trimming : {X_train_num.columns} ')
        logger.info(f'After Succeessful trimming : {X_test_num.columns}')

        return X_train_num,X_test_num


    except Exception as e:
        er_ty,er_msg,er_lin=sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

