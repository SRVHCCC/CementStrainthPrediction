from src.components.data_ingetion import DataIngetion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer
from src.logger import logging
from src.exception import CustomException

if __name__=='__main__':

    obj=DataIngetion()
    train_data_path,test_data_path=obj.initiate_data_ingetion()
    

    obj1=DataTransformation()
    train_data,test_data=obj1.initiate_data_tranformation(train_data_path,test_data_path)

    obj2=ModelTrainer()
    obj2.model_trainer_initiate(train_data,test_data)

