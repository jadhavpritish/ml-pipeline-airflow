import logging
import os

import opendatasets as od
import pandas as pd
import pendulum
from airflow.decorators import dag, task

logger = logging.getLogger(__name__)

# https://github.com/apache/airflow/discussions/24463
# https://stackoverflow.com/questions/75980623/why-is-my-airflow-hanging-up-if-i-send-a-http-request-inside-a-task
os.environ["NO_PROXY"] = "*"

DATASET_URL = (
    "https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification"
)


@dag(
    schedule=None,
    start_date=pendulum.today("UTC").add(days=-14),
    catchup=False,
    is_paused_upon_creation=False,
    tags=["mobile-price-classifier"],
)
def mobile_price_model_trainer():
    @task()
    def download_data():
        od.download(DATASET_URL)

    @task()
    def dummy():
        return None

    res2 = download_data()
    res1 = dummy()
    res1 >> res2


mobile_price_model_trainer()
