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
from sklearn.feature_selection import VarianceThreshold
logger =function('quasi_const')

def quasi_tech(train_num,test_num):
    try:

        var=VarianceThreshold(threshold=0.1)
        var.fit(train_num)
        logger.info(f'Total columns : {train_num.shape[1]} -> without variance 0 : {sum(var.get_support())} -> with Variance 0.1 : {sum(~var.get_support())}')
        logger.info(f'Variance 0.1 : names : {train_num.columns[~var.get_support()]}')

        train_num=train_num.drop(['RevolvingUtilizationOfUnsecuredLines_log_trim', 'age_log_trim'],axis=1)
        test_num=test_num.drop(['RevolvingUtilizationOfUnsecuredLines_log_trim', 'age_log_trim'],axis=1)

        logger.info(f'training columns After removing variance 0.1 by using constant technique are: {train_num.columns}')
        logger.info(f'testing columns After removing variance 0.1 by using constant technique are: {test_num.columns}')

        return train_num,test_num
    except Exception as e:
        er_ty,er_msg,er_lin=sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')




