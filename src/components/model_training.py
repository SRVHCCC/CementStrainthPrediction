import os
import sys
from src.components import data_transformation
from src.utils import save_object
from src.utils import model_evaluation
from src.logger import logging
from src.exception import CustomException
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass

@dataclass
class ModelTrainerconfig():
    model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer():
    def __init__(self):
        self.trainer_path=ModelTrainerconfig()

    def model_trainer_initiate(self,train_arr,test_arr):
        try:
           
            x_train=train_arr[:,:-1]
            y_train=train_arr[:,-1]
            x_test= test_arr[:,:-1]
            y_test= test_arr[:,-1]

            model,accuracy,model_name=model_evaluation(x_train,y_train,x_test,y_test)
            print(f"best model is : {model_name}  accuracy : {accuracy}")

            save_object(file_path=self.trainer_path.model_path,obj=model)

            



        except Exception as e:
            print(e)