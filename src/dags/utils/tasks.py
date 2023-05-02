from utils.helper import get_symbols_df, get_dataset, get_eng_dataset
from  utils import config
# define global variables


def data_processing():
    # get security symbols
    symbols_df = get_symbols_df(file_path=config.SYMBOLS_FILE_PATH)
    dataset_df = get_dataset(
        base_paths=config.BASE_PATH, 
        symbols_df=symbols_df
    )
    # Save dataset to parquet
    dataset_df.to_parquet(config.DATASET_PATH, index=False)

def data_engineering():
    # create engineered dataset
    eng_dataset = get_eng_dataset(path=config.DATASET_PATH)
    # save the engineered dataset
    eng_dataset.to_parquet(config.ENG_DATASET_PATH, index=False)
    return 