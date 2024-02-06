import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngetionconfig():
    train_data_path=os.path.join("artifacts","train.csv")
    test_data_path=os.path.join("artifacts","test.csv")
    row_data_path=os.path.join("artifacts","row.csv")

class DataIngetion():
    try:
        def __init__(self):
            self.data_ingetion_config=DataIngetionconfig()

        def initiate_data_ingetion(self):
            
            logging.info("data ingetion start")

            df=pd.read_csv(os.path.join("notebook/data","concrete_data.csv"))

            df.drop_duplicates(inplace=True,ignore_index=True)

            df = df.rename(columns=lambda x: x.strip())

            os.makedirs(os.path.dirname(self.data_ingetion_config.row_data_path),exist_ok=True)

            df.to_csv(self.data_ingetion_config.row_data_path,index=False)

            logging.info("train test split")

            x_train,x_test=train_test_split(df,test_size=0.31,random_state=43)

            x_train.to_csv(self.data_ingetion_config.train_data_path,index=False,header=True)

            x_test.to_csv(self.data_ingetion_config.test_data_path,index=False,header=True)
            
            return( self.data_ingetion_config.train_data_path,
                   self.data_ingetion_config.test_data_path
            )
        
        logging.info("data ingetion completed")






    except Exception as e:

        print(e)

