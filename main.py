import logging
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import sys
import warnings
warnings.filterwarnings('ignore')
from random_sample import random_values
from loggin import function
from transformation import log_transform
from trimming_tech import trimming_technique
from const_tech import constant_tech
from quasi_const import quasi_tech
from hypothesis import hypothesis_testing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from training import common
logger =function('main')

class CREDIT:
    try:
        def __init__(self,path):
            self.df=pd.read_csv(path)
            logger.info(f'Data loaded Successfull :{self.df.shape}')

            self.df=self.df.drop([150000,150001],axis=0)
            self.df=self.df.drop(['MonthlyIncome.1'],axis=1)
            logger.info(f'Data loaded Successfull :{self.df.isnull().sum()}')
            self.X=self.df.iloc[:,:-1]
            self.y=self.df.iloc[:,-1]
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f'Data loaded Successfull :{len(self.X_train),len(self.y_train)}')
    except Exception as e:
        er_ty,er_msg,er_lin=sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


    def missing_values(self):
        try:
            self.X_train,self.X_test=random_values(self.X_train,self.X_test)
            self.X_train=self.X_train.drop(['MonthlyIncome','NumberOfDependents'],axis=1)
            self.X_test=self.X_test.drop(['MonthlyIncome','NumberOfDependents'],axis=1)
            logger.info(self.X_train.isnull().sum())
            logger.info(self.X_test.isnull().sum())
            self.X_train_num=self.X_train.select_dtypes(exclude='object')
            self.X_train_cat=self.X_train.select_dtypes(include='object')
            self.X_test_num = self.X_test.select_dtypes(exclude='object')
            self.X_test_cat = self.X_test.select_dtypes(include='object')
            logger.info(f'Numirical columns are :{self.X_train_num.columns}')
            logger.info(f'Caterigerical Columns are : {self.X_train_cat.columns}')

        except Exception as e:
            er_ty,er_msg,er_lin=sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def handling_outliers(self):
        try:
            self.X_train_num,self.X_test_num=log_transform(self.X_train_num,self.X_test_num)
            logger.info(f'Tranformation completed Successfully : {self.X_train_num.columns}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def feature_selection(self):
        try:
            self.X_train_num, self.X_test_num = trimming_technique(self.X_train_num, self.X_test_num)
            logger.info(f'Trimming  completed Successfully : {self.X_train_num.columns}')

            # Applying filter methods
            # 1.Applying constant technique

            self.X_train_num, self.X_test_num = constant_tech(self.X_train_num, self.X_test_num)
            self.X_train_num, self.X_test_num = quasi_tech(self.X_train_num, self.X_test_num)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


    def hypo_testing(self):
        try:
            self.X_train_num,self.X_test_num=hypothesis_testing(self.X_train_num,self.X_test_num,self.y_train,self.y_test)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


    def cat_to_num(self):
        try:
            logger.info(f'Categorical Columns:')
            logger.info(f'Train Categorical Columns : {self.X_train_cat.columns}')
            logger.info(f'Test categorical Columns : {self.X_test_cat.columns}')

            #We are Applying OneHotEncoder to Gender and Region because both have same priority
            one_hot=OneHotEncoder(categories='auto',drop='first',handle_unknown='ignore')
            one_hot.fit(self.X_train_cat[['Gender','Region']])
            logger.info(one_hot.categories_)
            logger.info(one_hot.get_feature_names_out())
            res=one_hot.transform(self.X_train_cat[['Gender','Region']]).toarray()
            res_test=one_hot.transform(self.X_test_cat[['Gender','Region']]).toarray()
            f=pd.DataFrame(res,columns=one_hot.get_feature_names_out())
            f_test=pd.DataFrame(res_test,columns=one_hot.get_feature_names_out())
            self.X_train_cat.reset_index(drop=True,inplace=True)
            f.reset_index(drop=True,inplace=True)
            self.X_test_cat.reset_index(drop=True,inplace=True)
            f_test.reset_index(drop=True,inplace=True)
            self.X_train_cat=pd.concat([self.X_train_cat,f],axis=1)
            self.X_test_cat=pd.concat([self.X_test_cat,f_test],axis=1)

            logger.info(self.X_train_cat.isnull().sum())
            logger.info(self.X_test_cat.isnull().sum())

            logger.info(f'Sample Training Data : {self.X_train_cat.columns}-------->{self.X_train_cat.sample(5)}')
            logger.info(f'Sample Test Data : {self.X_test_cat.columns}-------->{self.X_test_cat.sample(5)}')



            ##We are Applying Ordinal Encoder to Rented_house | Education |  Occupation because these treated based on priority

            #before applying ordinal encoder
            logger.info(f'before applying Ordinal encoder training Occupation: {self.X_train_cat['Occupation'].head(10)}')
            logger.info(f'before applying Ordinal encoder training Education: {self.X_train_cat['Education'].head(10)}')
            logger.info(f'after applying Ordinal encoder training Region : {self.X_train_cat['Region_East'].head(1)}--{self.X_train_cat['Region_West'].head(1)}---{self.X_train_cat['Region_North'].head(1)} - ---{self.X_train_cat['Region_South'].head(1)}')

            logger.info((f'before applyiing '))
            ordinal = OrdinalEncoder()
            ordinal.fit(self.X_train_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
            logger.info(ordinal.categories_)
            logger.info(ordinal.get_feature_names_out())
            res1 = ordinal.transform(self.X_train_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
            res_test1 = ordinal.transform(self.X_test_cat[['Rented_OwnHouse', 'Occupation', 'Education']])
            f1 = pd.DataFrame(res1, columns=ordinal.get_feature_names_out()+['_conv'])
            f_test1 = pd.DataFrame(res_test1, columns=ordinal.get_feature_names_out()+['_conv'])
            self.X_train_cat.reset_index(drop=True, inplace=True)
            f1.reset_index(drop=True, inplace=True)
            self.X_test_cat.reset_index(drop=True, inplace=True)
            f_test1.reset_index(drop=True, inplace=True)
            self.X_train_cat = pd.concat([self.X_train_cat, f1], axis=1)
            self.X_test_cat = pd.concat([self.X_test_cat, f_test1], axis=1)

            #dropping same column values

            self.X_train_cat=self.X_train_cat.drop(['Gender', 'Region','Rented_OwnHouse', 'Occupation', 'Education'],axis=1)
            self.X_test_cat=self.X_test_cat.drop(['Gender', 'Region','Rented_OwnHouse', 'Occupation', 'Education'],axis=1)

            logger.info(self.X_train_cat.isnull().sum())
            logger.info(self.X_test_cat.isnull().sum())

            logger.info(f'after applying Ordinal encoder training Occupation : {self.X_train_cat['Occupation_conv'].head(10)}')
            logger.info(f'after applying Ordinal encoder training Education  : {self.X_train_cat['Education_conv'].head(10)}')
            logger.info(f'after applying Ordinal encoder training Region : {self.X_train_cat['Region_East'].head(1)}--{self.X_train_cat['Region_West'].head(1)}---{self.X_train_cat['Region_North'].head(1)}----{self.X_train_cat['Region_South'].head(1)}')



            logger.info(f'Sample Training Data : {self.X_train_cat.columns}-------->{self.X_train_cat.sample(5)}')
            logger.info(f'Sample Test Data : {self.X_test_cat.columns}-------->{self.X_test_cat.sample(5)}')

            #Dependent variable can be done by using LabelEncoder
            logger.info(f'y_train_data : {self.y_train.unique()}')
            logger.info(f'y_train_data : {self.y_train.isnull().sum()}')
            logger.info(f'y_test_data : {self.y_test.unique()}')
            logger.info(f'y_test_data : {self.y_test.isnull().sum()}')
            # dependent varibale should be converted using label encoder
            logger.info(f'{self.y_train[:10]}')
            lb = LabelEncoder()
            lb.fit(self.y_train)
            self.y_train = lb.transform(self.y_train)
            self.y_test = lb.transform(self.y_test)
            logger.info(f'detailed : {lb.classes_} ')
            logger.info(f'{self.y_train[:10]}')
            logger.info(f'y_train_data : {self.y_train.shape}')
            logger.info(f'y_test_data : {self.y_test.shape}')

            # 0 -> Bad
            # 1 -> Good

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def merge_data(self):
        try:
            self.X_train_num.reset_index(drop=True,inplace=True)
            self.X_train_cat.reset_index(drop=True,inplace=True)

            self.X_test_num.reset_index(drop=True,inplace=True)
            self.X_test_cat.reset_index(drop=True,inplace=True)

            self.training_data=pd.concat([self.X_train_num,self.X_train_cat],axis=1)
            self.testing_data=pd.concat([self.X_test_num,self.X_test_cat],axis=1)

            logger.info(f'Training data : {self.training_data.columns}')
            logger.info(f'Testing data : {self.testing_data.columns}')

        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def data_balancing(self):
        try:
            logger.info('--------------------Before Balancing-------------------------')
            logger.info(f'Number of Good category for total training data {self.training_data.shape[0]} is : {sum(self.y_train==1)}')
            logger.info(f'Number of Bad category for total training data {self.training_data.shape[0]} is : {sum(self.y_train == 0)}')

            sm=SMOTE(random_state=42)
            self.training_data_res,self.y_train_res=sm.fit_resample(self.training_data,self.y_train)

            logger.info('--------------------After Balancing-------------------------')
            logger.info(f'Number of Good category for total training data {self.training_data_res.shape[0]} is : {sum(self.y_train_res == 1)}')
            logger.info(f'Number of Bad category for total training data {self.training_data_res.shape[0]} is : {sum(self.y_train_res == 0)}')
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

    def feature_scaling(self):
        try:
            logger.info('---------------Before Scaling------------')
            logger.info(f'Shape is : {self.training_data_res.shape}--------{self.training_data_res.columns}')
            logger.info(self.training_data_res.head(4))
            stand=StandardScaler()
            stand.fit(self.training_data_res)
            self.training_data_res_t=stand.transform(self.training_data_res)
            self.testing_data_t=stand.transform(self.testing_data)
            logger.info('------------After Scaling-------------------------')
            logger.info(self.training_data_res_t)
            logger.info(self.testing_data_t)
            logger.info(self.training_data_res.columns)

            #saving Standard scaler also
            with open('standard_scalar.pkl', 'wb') as t:
                pickle.dump(stand, t)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


    def train_models(self):
        try:
            common(self.training_data_res_t, self.y_train_res, self.testing_data_t, self.y_test)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')

        #we finalized the model with Logistic Regression by AUC-ROC

    def best_model_auc_and_roc(self):
        try:
            self.reg_lr=LogisticRegression()
            self.reg_lr.fit(self.training_data_res_t,self.y_train_res)
            logger.info(f'Accuracy Score is : {accuracy_score(self.y_test,self.reg_lr.predict(self.testing_data))}')
            logger.info(f'Confusion Matxis  is :\n {confusion_matrix(self.y_test, self.reg_lr.predict(self.testing_data))}')
            logger.info(f'Classification Report  is : \n{classification_report(self.y_test, self.reg_lr.predict(self.testing_data))}')


            #Saving Model

            with open('creditcard.pkl','wb') as f:
                pickle.dump(self.reg_lr,f)
        except Exception as e:
            er_ty, er_msg, er_lin = sys.exc_info()
            logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')








if __name__=='__main__':
    try:
        path='C:\\Users\\nikhi\\Downloads\\Credit Card Project\\creditcard.csv'
        obj=CREDIT(path)
        obj.missing_values()
        obj.handling_outliers()
        obj.feature_selection()
        obj.hypo_testing()
        obj.cat_to_num()
        obj.merge_data()
        obj.data_balancing()
        obj.feature_scaling()
        obj.train_models()
        obj.best_model_auc_and_roc()

    except Exception as e:
        er_ty,er_msg,er_lin=sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')


