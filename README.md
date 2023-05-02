# etf-stocks

This project is a sample for demonstration of silks need as a Data Engineer and/or a Machine Learning Engineer.

Models trained in this project are deployed at `https://etf-stock.onrender.com/`. 

Swagger documentation for the APIs is available at `https://etf-stock.onrender.com/api/docs`. 

Prediction API is available at `https://etf-stock.onrender.com/api/predict`. The api takes in 3 arguments 
* vol_moving_avg: float [required]
* adj_close_rolling_med: float [required]
* model_type: string [optional, default='Random Forest']

## Dependencies
* Docker
* docker-compose
* Python3.9
* Jupyter Notebooks
* Airflow
* Torch
* Scikit-learn
* Pandas
* Numpy
* Dask
* FastAPI

## Installation 
1. Clone the repo
```
git clone https://github.com/AbdulMutakabbir/etf-stocks.git
```
2. Enter into the repo
```
cd etf-stocks
```
3. Deploy the APIs:
```
docker compose -f docker-compose.api.yml up --build
```
4. Get the dataset:

Dataset url: https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

Extract the dataset zip file and store the data as shown in the [tree structure](https://github.com/AbdulMutakabbir/etf-stocks#project-structure).

5. Deploy Airflow:
```
docker compose  -f docker-compose.airflow.yml up --build 
```
6. Explore the notebooks
```
pip install requirements.txt
```
Launch the jupyter server and explore the [notebooks](https://github.com/AbdulMutakabbir/etf-stocks/tree/main/notebooks).

## Project Structure
> This repos is structured as follows: 
> ```
> etf-stocks
> ├── README.md
> ├── LICENSE
> ├── .env                          # Docker environment variables 
> ├── .gitignore
> ├── render.yaml                   # script to deploy APIs on render
> ├── config.tomli                  # data processing and modeling configurations (used in notebooks)
> ├── requirements.txt
> ├── requirements_api.txt
> ├── requirements_airflow.txt
> ├── DockerFile.api 
> ├── DockerFile.airflow
> ├── docker-compose.api.yml
> ├── docker-compose.airflow.yml
> ├── data
> │   ├── symbols_valid_meta.csv    # should be extracted form kaggle
> │   ├── etf                       # should be extracted form kaggle
> │   │   └ <dataset extracted from kaggle for ETF should be stored here>
> │   ├── Stocks                    # extracted form kaggle
> │   │   └ <dataset extracted from kaggle for Stocks should be stored here>
> │   ├── processed
> │   │   └ <stores processed dataset (Stage 1) here>
> │   └── engineered 
> │       └ <stores engineered dataset (stage 2) here>
> ├── notebooks                     # divided in stages for easy access 
> │   └ <stores jupyter notebooks for data extraction, modeling, and exploration>
> ├── src
> │   └── dags                      # contains the code for Airflow DAG(s)
> │       ├── utils
> │       │   ├── __init__.py
> │       │   ├── config.py         # contains config for Airflow DAG(s)
> │       │   ├── helper.py
> │       │   ├── model_trining_deep_learning.py
> │       │   ├── model_trining_random_forest.py
> │       │   └── tasks.py          # PythonOperator code for Airflow DAG(s)
> │       ├── .airflowignore
> │       └── data_pipeline.py      # main file for running the pipeline
> ├── models                        # stores model data 
> │   ├── dataset_stats_deep_learning.joblib
> │   ├── model_deep_learning.joblib
> │   └── model_random_forest.joblib
> ├── logs
> │   ├── deep_learning.log
> │   ├── random_forest.log
> │   └── <airflow logs>
> ├── api                           # contains code for Fast API Deployment
> │   ├── model                     # contains code for API data structure
> │   ├── routes                    # contains code for API routes
> │   ├── utils                     # contains code helper code for APIs
> │   └── app.py                    # main file to start FastAPI server
> └── test
> ```
>  

## DAG Structure

![DAG structure](https://raw.githubusercontent.com/AbdulMutakabbir/etf-stocks/09c13ea85e3c1d7fb9df6c253a1987466eabe069/assets/dag_structure.png)
```
- Stage 0 (Dependency Installation): Install the python requirements if not installed
- Stage 1 (Raw Data Processing): Data is extracted from the raw fils and stored into 
- Stage 2 (Data Engineering): Feature engineering is performed on the data 
- Stage 3 (Model Training):
  - Stage 3.1: Train Random Forest Model
  - Stage 3.2: Train Deep Learning Model
```
   
