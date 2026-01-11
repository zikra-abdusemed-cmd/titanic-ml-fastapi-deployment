from fastapi import FastAPI
import pandas as pd
import joblib
from app.schemas import PassengerData

app = FastAPI()

logreg_model = joblib.load("models/titanic_logistic_model.joblib")
dt_model = joblib.load("models/titanic_decision_tree_model.joblib")

@app.get("/")
def root():
    return {"message": "Titanic ML API is running"}

def prepare_input(data: PassengerData):
    return pd.DataFrame([{
        "Pclass": data.Pclass,
        "Sex": data.Sex,
        "Age": data.Age,
        "SibSp": data.SibSp,
        "Parch": data.Parch,
        "Fare": data.Fare,
        "Embarked": data.Embarked
    }])

@app.post("/predict/logistic")
def predict_logistic(data: PassengerData):
    df = prepare_input(data)
    prediction = logreg_model.predict(df)[0]
    return {"survived": int(prediction)}

@app.post("/predict/tree")
def predict_tree(data: PassengerData):
    df = prepare_input(data)
    prediction = dt_model.predict(df)[0]
    return {"survived": int(prediction)}
