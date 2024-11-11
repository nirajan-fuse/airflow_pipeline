import os
from datetime import datetime, timedelta
import pendulum

# from airflow import DAG
# from airflow.operators.python import PythonOperator
# from airflow.sensors.python import PythonSensor

from airflow.decorators import task, dag

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

DATA_PATH = "./data/Heart Attack.csv"
MODEL_PATH = "./model/model.pkl"
THRESHOLD = 0.90

default_args = {
    "depends_on_past": False,
    "email": ["nirajan.thakuri@fusemachines.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# with DAG(
#     dag_id="heart_disease_prediction_pipeline",
#     default_args=default_args,
#     description="A simple DAG pipeline.",
#     schedule=timedelta(days=1),
#     start_date=datetime(2024, 11, 9),
#     tags=["pipeline"],
# ) as dag:

# preprocess_task = PythonOperator(
#     task_id="preprocess_data", python_callable=preprocess_data
# )

# train_task = PythonOperator(task_id="train_model", python_callable=train_model)

# check_model_sensor = PythonSensor(
#     task_id="check_new_model",
#     python_callable=check_new_model,
#     mode="reschedule",
#     poke_interval=60,
#     timeout=3600,
# )

# inference_task = PythonOperator(task_id="inference", python_callable=inference)


@dag(
    schedule=timedelta(days=1),
    start_date=pendulum.datetime(2024, 11, 9, tz="UTC"),
    default_args=default_args,
    description="A simple DAG pipeline.",
    catchup=False,
    tags=["pipeline"],
)
def heart_disease_prediction_pipeline():

    @task
    def preprocess_data():
        data = pd.read_csv(DATA_PATH)
        data.dropna(inplace=True)
        df = data[data["impluse"] < 1000]
        if df["class"].dtypes == "object":
            label = {"positive": 1, "negative": 0}
            df["class"] = df["class"].map(label)
        X = df.drop(columns="class")
        y = df["class"]
        # ti.xcom_push(key="X", value=X)
        # ti.xcom_push(key="y", value=y)
        X.to_csv("./data/X.csv", index=False)
        y.to_csv("./data/y.csv", index=False)
        print("X and y saved to /data directory")

    @task
    def train_model():
        # X = ti.xcom_pull(key="X", task_ids="preprocess_data")
        # y = ti.xcom_pull(key="y", task_ids="preprocess_data")
        X = pd.read_csv("./data/X.csv")
        y = pd.read_csv("./data/y.csv").squeeze()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy >= THRESHOLD:
            with open(MODEL_PATH, "wb") as file:
                pickle.dump(model, file)
            print(f"Model saved with accuracy: {accuracy}")
        else:
            raise ValueError(f"Model accuracy ({accuracy}) is below threshold")

    @task.sensor(poke_interval=60, timeout=3600, mode="reschedule")
    def check_new_model():
        return os.path.exists(MODEL_PATH)

    @task
    def inference():
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        X = pd.read_csv("./data/X.csv")
        # X = ti.xcom_pull(key="X", task_ids="preprocess_data")

        predictions = model.predict(X)
        print(predictions)

    data_prep = preprocess_data()
    model_training = train_model()
    model_check = check_new_model()
    inference_task = inference()

    data_prep >> model_training >> model_check >> inference_task


heart_disease_prediction_pipeline()
