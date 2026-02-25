from fastapi import APIRouter, status
import pandas as pd
from api.models import Student
from api.schemas import PredictStudent
from fastapi import HTTPException
import traceback

from api.database import SessionLocal
from src.model_builder import ModelBuilder
import logging
from datetime import datetime
import json

logging.basicConfig(
    filename='logs/model_monitoring.log',
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)

router = APIRouter(
    tags=["student"],
)

@router.post(
    path="/predict/",
    summary="Predicting the student's school stage",
    response_description="Predicted class",
    status_code=status.HTTP_200_OK,
)
def predict_student(schema: PredictStudent):

    try:
        df = pd.DataFrame([schema.model_dump()])
        df["fase"] = ""
        predictor = ModelBuilder(df)
        prediction = predictor.predict(df)

        df["classe_defas"] = prediction
        df.drop(columns="fase").to_sql("student", con=SessionLocal().bind, if_exists="append", index=False)
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": schema.model_dump(),
            "prediction": prediction[0],
            "model_version": "v2.0"
        }
        logging.info(json.dumps(log_entry))
        
        return {"prediction": prediction}

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))