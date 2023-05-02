import pandas as pd
import numpy as np
from tqdm import trange, tqdm
import os
import logging

def get_logger():
    # Create a logger object
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a file handler and set the logging level
    file_handler = logging.FileHandler('/opt/airflow/logs/DAG_airflow.log')
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger

def get_file_name(index:int, file_list:list):
    return file_list[index]

def get_file_path(file_name, base_path:str):
    return f"{base_path}/{file_name}"

def get_symbol(file_name:str):
    return file_name.split(".")[0]

def get_security_name(symbol:str, symbols_df:pd.DataFrame):
    try:
        security_name = symbols_df.loc[symbol]['Security Name'] 
        return security_name
    except KeyError:
        return None  
    
def get_symbols_df(file_path:str):
    # load symbols info and index for future query
    symbols_df = pd.read_csv(
        file_path, 
        index_col='Symbol'
    )
    # drop other columns to conserve space
    symbols_df.drop(
        [
            'Nasdaq Traded', 
            'Listing Exchange',
            'Market Category',
            'Round Lot Size',
            'Test Issue',
            'Financial Status',
            'CQS Symbol',
            'NASDAQ Symbol',
            'NextShares'
        ],
        axis=1,
        inplace=True
    )
    return symbols_df

def get_security_df(file_path:str, symbol:str, security_name:str, date_format:str='%Y-%m-%d')->pd.DataFrame:
    # read file
    security_df = pd.read_csv(file_path, parse_dates=['Date']) 
    # parse date format                      
    security_df['Date'] = security_df['Date'].dt.strftime(date_format)  
    # dropping null or inf values
    security_df = security_df.replace([np.inf, -np.inf], np.nan).dropna()   
    # set datatypes
    security_df['Open'] = security_df['Open'].astype(float)
    security_df['High'] = security_df['High'].astype(float)
    security_df['Low'] = security_df['Low'].astype(float)
    security_df['Close'] = security_df['Close'].astype(float)
    security_df['Adj Close'] = security_df['Adj Close'].astype(float)
    security_df['Volume'] = security_df['Volume'].round().astype(int)    
    # Add additional columns
    # note:     
    # This can be stored in a RDBMS format (SQL) 
    # which will save the redundant column space ('Symbol', 'Security Name') 
    # Since parquet is suggested, it was not considered 
    security_df['Symbol'] = symbol                                                      
    security_df['Security Name'] = security_name    
    return security_df                      

def get_dataset(base_paths:list, symbols_df:pd.DataFrame):
    # init local variables 
    dataset_dfs = []

    # loop over all base paths 
    for base_path in base_paths:
        print(f"Started extraction for directory {base_path}")
        files = sorted(os.listdir(base_path))
        file_count = len(files)
        # loop over all securities in the folder
        for file_index in trange(file_count, unit="file"):
            file_name = get_file_name(
                index=file_index,
                file_list=files
            )
            file_path = get_file_path(
                file_name=file_name, 
                base_path=base_path
            )
            symbol = get_symbol(file_name=file_name)
            security_name = get_security_name(
                symbol=symbol, 
                symbols_df=symbols_df
            )
            # if security info is found continue
            if security_name is not None:
                security_df = get_security_df(
                    file_path=file_path, 
                    symbol=symbol, 
                    security_name=security_name
                )
                dataset_dfs.append(security_df) 
    
    # merge all securities data
    dataset_df = pd.concat(dataset_dfs)
    del dataset_dfs
    return dataset_df

def get_featured_dataset(security_df:pd.DataFrame)->pd.DataFrame:
    # calculates the rolling median and average
    security_df = security_df.set_index('Date')
    security_df = security_df.resample('D').ffill()
    security_df['vol_moving_avg'] = security_df['Volume'].rolling(window=30).mean()
    security_df['adj_close_rolling_med'] = security_df['Adj Close'].rolling(window=30).median()
    security_df = security_df.reset_index()
    return security_df

def read_dataset_data_time(path:str, date_format:str='%Y-%m-%d'):
    # read dataset
    dataset = pd.read_parquet(path)
    # to datetime
    dataset['Date'] = pd.to_datetime(dataset['Date'], format=date_format)
    return dataset

def get_eng_dataset(path:str):
    dataset = read_dataset_data_time(path=path)
    
    # get unique securities
    securities =  dataset['Symbol'].unique()

    security_dfs = []

    # loop over all securities to obtain the feature dataset
    for security in tqdm(securities):
        # get data for security
        security_df = dataset[dataset['Symbol'] == security]
        # get featured dataset
        security_df = get_featured_dataset(security_df=security_df)
        # append the security df
        security_dfs.append(security_df)

    # merge all dfs to create engineered dataset 
    eng_dataset = pd.concat(security_dfs)
    del security_dfs

    # discarding null values 
    eng_dataset = eng_dataset.dropna()
    return eng_dataset
