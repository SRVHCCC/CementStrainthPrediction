import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationconfig():
    preprocesor_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation():
    def __init__(self):
        self.preprocesor_config=DataTransformationconfig()


    def DataPreprocessor(self):
        try:
            preprocessor=Pipeline(
            steps=[("inputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())])
            
            return preprocessor
            
        except Exception as e:
            print(e)


    def initiate_data_tranformation(self,train_data_path,test_data_path):
        try:
            logging.info("Data Transformation Start : ")

            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)

            preprocesor_obj=self.DataPreprocessor()
            

            target_column="concrete_compressive_strength"
            drop_column=[target_column]

            x_train_data=train_data.drop(columns=drop_column)
            y_train_data=train_data[target_column]
            x_test_data=test_data.drop(columns=drop_column)
            y_test_data=test_data[target_column]

            x_train=preprocesor_obj.fit_transform(x_train_data)
            x_test=preprocesor_obj.transform(x_test_data)

            save_object(file_path=self.preprocesor_config.preprocesor_path,obj=preprocesor_obj)

            train_arr=np.c_[x_train,np.array(y_train_data)]
            test_arr=np.c_[x_test,np.array(y_test_data)]

            return(
                train_arr,
                test_arr,
                
            )

            
        except Exception as e:
            print(e)