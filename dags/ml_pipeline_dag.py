import logging
import os
from typing import TypedDict

import joblib
import opendatasets as od
import pandas as pd
import pendulum
from airflow.decorators import dag, task
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DataSplit(TypedDict):
    train_x_fp: str
    train_y_fp: str
    val_x_fp: str
    val_y_fp: str


class TransformerOut(TypedDict):
    pipeline_fp: str


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
    def download_data() -> str:
        od.download(DATASET_URL)
        return DATASET_URL.split("/")[-1]

    @task()
    def train_validation_split(dataset_folder_name: str) -> DataSplit:
        raw_df = pd.read_csv(f"./{dataset_folder_name}/train.csv", sep=",")
        raw_y = raw_df.pop("price_range")
        train_x, val_x, train_y, val_y = train_test_split(raw_df, raw_y, train_size=0.8)

        train_x.to_csv(f"./{dataset_folder_name}/train_x.csv", index=False, sep=",")
        train_y.to_csv(f"./{dataset_folder_name}/train_y.csv", index=False, sep=",")
        val_x.to_csv(f"./{dataset_folder_name}/val_x.csv", index=False, sep=",")
        val_y.to_csv(f"./{dataset_folder_name}/val_y.csv", index=False, sep=",")

        return DataSplit(
            train_x_fp=f"./{dataset_folder_name}/train_x.csv",
            train_y_fp=f"./{dataset_folder_name}/train_y.csv",
            val_x_fp=f"./{dataset_folder_name}/val_x.csv",
            val_y_fp=f"./{dataset_folder_name}/val_y.csv",
        )

    @task()
    def train_transformation_pipeline(train_x_fp: str) -> Pipeline:

        logger.debug(f"Reading training data: {train_x_fp}")
        train_x = pd.read_csv(train_x_fp, sep=",")

        # Train Sklearn pipeline for preprocessing train_x
        pipeline = Pipeline([("scaler", StandardScaler())])
        pipeline.fit(train_x)
        pipeline_fp = "./pipeline.pkl"
        joblib.dump(pipeline, pipeline_fp)

        return TransformerOut(pipeline_fp=pipeline_fp)

    @task()
    def train_classifier(datasplit: DataSplit, transformer: TransformerOut):

        train_x = pd.read_csv(datasplit["train_x_fp"], sep=",")
        train_y = pd.read_csv(datasplit["train_y_fp"], sep=",")
        val_x = pd.read_csv(datasplit["val_x_fp"], sep=",")
        val_y = pd.read_csv(datasplit["val_y_fp"], sep=",")

        pipeline = joblib.load(transformer["pipeline_fp"])

        transformed_train_x = pipeline.transform(train_x)
        transformed_val_x = pipeline.transform(val_x)

        clf = LogisticRegression(random_state=0).fit(transformed_train_x, train_y)

        val_score = clf.score(transformed_val_x, val_y)
        logger.info(f"Score: {val_score}")
        return val_score

    dataset_folder_name = download_data()
    datasplit = train_validation_split(dataset_folder_name=dataset_folder_name)
    transformer = train_transformation_pipeline(train_x_fp=datasplit["train_x_fp"])
    res = train_classifier(datasplit, transformer)

    dataset_folder_name >> datasplit >> transformer
    datasplit >> res
    transformer >> res


mobile_price_model_trainer()
