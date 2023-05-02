from datetime import timedelta
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

from utils.tasks import data_processing, data_engineering

# init args for DAG
default_args = {
    'Owner': 'airflow',
    'start_date': days_ago(1)
}

# create data pipeline dag
data_pipeline_dag = DAG(
    'data_pipeline_dag',
    default_args = default_args,
    description = "This DAG does the ETL and training of the models.",
    schedule_interval = None, # We dont want it to repeat
    catchup = False
)

# create tasks 
dependency_installation_task = BashOperator(
    task_id ='dependency_installation_task',
    bash_command ='''su - airflow -c "pip install pandas numpy tqdm pyarrow fastparquet scikit-learn joblib torch"''',
    dag = data_pipeline_dag
)

task_1_raw_data_processing = PythonOperator(
    task_id = 'task_1_raw_data_processing',
    python_callable = data_processing,
    dag = data_pipeline_dag
)

task_2_data_engineering = PythonOperator(
    task_id = 'task_2_data_engineering',
    python_callable = data_engineering,
    dag = data_pipeline_dag
)

task_3_1_train_RF = BashOperator(
    task_id ='task_3_1_train_RF',
    bash_command ='cd /opt/airflow/src/dags && python3 model_training_random_forest.py',
    dag = data_pipeline_dag
)

task_3_2_train_DL = BashOperator(
    task_id ='task_3_2_train_DL',
    bash_command ='cd /opt/airflow/src/dags && python3 model_training_deep_learning.py',
    dag = data_pipeline_dag
)

# define task structure
dependency_installation_task >> task_1_raw_data_processing >> task_2_data_engineering >> [task_3_1_train_RF, task_3_2_train_DL]