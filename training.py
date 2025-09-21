import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sys
import warnings
warnings.filterwarnings('ignore')
from loggin import function
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

logger = function('training')

def knn_algo(X_train, y_train, X_test, y_test):
    try:
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        logger.info('-------------------KNN Algorithm-------------------')
        logger.info(accuracy_score(y_test, model.predict(X_test)))
        logger.info(confusion_matrix(y_test, model.predict(X_test)))
        global knn_pred
        knn_pred=model.predict(X_test)
        logger.info(classification_report(y_test, model.predict(X_test)))
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def lg_algo(X_train, y_train, X_test, y_test):
    try:
        model = LogisticRegression()
        model.fit(X_train, y_train)
        logger.info('-------------------Logistic Regression -------------------')
        logger.info(accuracy_score(y_test, model.predict(X_test)))
        logger.info(confusion_matrix(y_test, model.predict(X_test)))
        global lg_pred
        lg_pred = model.predict(X_test)
        logger.info(classification_report(y_test, model.predict(X_test)))
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def nb_algo(X_train, y_train, X_test, y_test):
    try:
        model = GaussianNB()
        model.fit(X_train, y_train)
        logger.info('-------------------Naive Bayes Algorithm-------------------')
        logger.info(accuracy_score(y_test, model.predict(X_test)))
        logger.info(confusion_matrix(y_test, model.predict(X_test)))
        global nb_pred
        nb_pred = model.predict(X_test)
        logger.info(classification_report(y_test, model.predict(X_test)))
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def dt_algo(X_train, y_train, X_test, y_test):
    try:
        model = DecisionTreeClassifier(criterion='entropy')
        model.fit(X_train, y_train)
        logger.info('-------------------Decision Tree Algorithm-------------------')
        logger.info(accuracy_score(y_test, model.predict(X_test)))
        logger.info(confusion_matrix(y_test, model.predict(X_test)))
        global dt_pred
        dt_pred = model.predict(X_test)
        logger.info(classification_report(y_test, model.predict(X_test)))
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

def rf_algo(X_train, y_train, X_test, y_test):
    try:
        model = RandomForestClassifier(n_estimators=7, criterion='entropy')
        model.fit(X_train, y_train)
        logger.info('-------------------Random Forest Algorithm-------------------')
        logger.info(accuracy_score(y_test, model.predict(X_test)))
        logger.info(confusion_matrix(y_test, model.predict(X_test)))
        global rf_pred
        rf_pred = model.predict(X_test)
        logger.info(classification_report(y_test, model.predict(X_test)))
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

'''def svm_algo(X_train, y_train, X_test, y_test):
    try:
        model = SVC(kernel='rbf')
        model.fit(X_train, y_train)
        logger.info('-------------------SVM Algorithm-------------------')
        logger.info(accuracy_score(y_test, model.predict(X_test)))
        logger.info(confusion_matrix(y_test, model.predict(X_test)))
        global svm_prd
        svm_pred=model.predict(X_test)
        logger.info(classification_report(y_test, model.predict(X_test)))
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')'''

     #Selecting best model Using AUC-ROC curve
def best_model(X_train, y_train, X_test, y_test):
    try:
        knn_fpr,knn_tpr,knn_thre=roc_curve(y_test,knn_pred)
        lg_fpr,lg_tpr,lg_thre=roc_curve(y_test,lg_pred)
        nb_fpr,nb_tpr,nb_thre=roc_curve(y_test,nb_pred)
        dt_fpr,dt_tpr,dt_thre=roc_curve(y_test,dt_pred)
        rf_fpr,rf_tpr,rf_thre=roc_curve(y_test,rf_pred)

        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('AUC-ROC Curves of All Models:')
        plt.plot([0,1],[0,1], "k--")


        plt.plot(knn_fpr,knn_tpr,color='r',label='KNN Algorithm')
        plt.plot(lg_fpr,lg_tpr,color='blue',label='Logistic Regression')
        plt.plot(nb_fpr,nb_tpr,color='green',label='Naive Bayes Algorithm')
        plt.plot(dt_fpr,dt_tpr,color='black',label='decision tree Algorithm')
        plt.plot(rf_fpr,rf_tpr,color='yellow',label='Rondom forest Algorithm')

        plt.legend(loc=0)
        plt.show()

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')





def common(X_train, y_train, X_test, y_test):
    try:
        knn_algo(X_train, y_train, X_test, y_test)
        lg_algo(X_train, y_train, X_test, y_test)
        nb_algo(X_train, y_train, X_test, y_test)
        dt_algo(X_train, y_train, X_test, y_test)
        rf_algo(X_train, y_train, X_test, y_test)
        #svm_algo(X_train, y_train, X_test, y_test)
        logger.info('AUC - ROC Curve For All Models:')
        best_model(X_train, y_train, X_test, y_test)
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')
