import os
import sys
import pickle
from src.logger import logging
from src.exception import CustomException
import dill
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score,mean_absolute_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def model_evaluation(x_train,y_train,x_test,y_test):
    try:
        dict1={"linearRegression":LinearRegression(),
       "ridge":Ridge(),
       "lasso":Lasso(),
       "decisionTreeRegressor":DecisionTreeRegressor()}
        model_accuracy={}

        for i in dict1:
            model1=dict1[i]
            model1.fit(x_train,y_train)
            y_pred=model1.predict(x_test)
            score=r2_score(y_test,y_pred)*100
            model_accuracy[i]=score
        print(model_accuracy)
        max1=max(model_accuracy,key=model_accuracy.get)
        model_name=max1
        accuracy=model_accuracy[max1]
        model=dict1[max1]
        return model,accuracy,model_name


    except Exception as e:
        print(e)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)