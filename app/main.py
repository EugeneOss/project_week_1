from fastapi import FastAPI, Request, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from catboost import CatBoostClassifier
import json
import numpy as np

app = FastAPI()

with open("./models/metrics_catboost.json", "r", encoding="utf-8") as f:
    metrics_catboost = json.load(f)

with open("./models/metrics_rfc.json", "r", encoding="utf-8") as f:
    metrics_rfc = json.load(f)

#Счетчики запросов
request_count_rfc_true = 0
request_count_cbc_true = 0
request_count_rfc_false = 0
request_count_cbc_false = 0


#Загружаем модели
model_cbc = CatBoostClassifier()
model_cbc.load_model("./models/best_model_catboost.cbm")
model_rfc = joblib.load("./models/random_forest_model.pkl")
features = model_cbc.feature_names_

@app.get('/stats')
def stats():
    return {"request_count_rfc_true": f'{request_count_rfc_true}',
            'request_count_cbc_true': f'{request_count_cbc_true}',
            'request_count_rfc_false': f'{request_count_rfc_false}',
            'request_count_cbc_false': f'{request_count_cbc_false}'
            }

@app.get("/")
def read_root():
    return {"message": "🚀 FastAPI работает! Добро пожаловать на API."}

@app.get('/health')
def health():
    return {'status': 'OK'}

@app.get('/predict_cbc')
def predict_cbc():
    global request_count_cbc_true
    global request_count_cbc_false
    global features
    global metrics_catboost
    global model_cbc

    test_df = pd.read_parquet('./data/test.parquet').sample(1)
    for_pred_X, for_pred_y = test_df[features], test_df['target']
    y_proba = model_cbc.predict_proba(for_pred_X)[:, 1]
    threshold = metrics_catboost['threshold']
    test_with_trash = np.where(y_proba > threshold, 1, 0)

    if test_with_trash[0] == for_pred_y.values[0]:
        request_count_cbc_true += 1
    else:
        request_count_cbc_false += 1

    return {
        'Model': 'CatBoostClassifier',
        'Предсказанное значение моделью': f'{test_with_trash[0]}',
        'Вероятность предсказания': f'{y_proba[0]:.4f}',
        'Трешхолд для модели': f'{threshold:.4f}',
        'Истинное значение': f'{for_pred_y.values[0]}',
        'Предсказание модели': "Верное" if test_with_trash[0] == for_pred_y.values[0] else "Неверно"
    }


@app.get('/predict_rfc')
def predict_rfc():
    global request_count_rfc_true
    global request_count_rfc_false
    global features
    global metrics_rfc
    global model_rfc

    test_df = pd.read_parquet('./data/test.parquet').sample(1)
    for_pred_X, for_pred_y = test_df[features], test_df['target']
    y_proba = model_rfc.predict_proba(for_pred_X)[:, 1]
    threshold = metrics_rfc['threshold']
    test_with_trash = np.where(y_proba > threshold, 1, 0)

    if test_with_trash[0] == for_pred_y.values[0]:
        request_count_rfc_true += 1
    else:
        request_count_rfc_false += 1

    return {
        'Model': 'RandomForestClassifier',
        'Предсказанное значение моделью': f'{test_with_trash[0]}',
        'Вероятность предсказания': f'{y_proba[0]:.4f}',
        'Трешхолд для модели': f'{threshold:.4f}',
        'Истинное значение': f'{for_pred_y.values[0]}',
        'Предсказание модели': "Верное" if test_with_trash[0] == for_pred_y.values[0] else "Неверно"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5000)

