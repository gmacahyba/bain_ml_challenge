from fastapi import APIRouter, Depends
from . import auth
from . import features
import pickle
import pandas as pd
import logging

logging.basicConfig(filename = "./api/model_serving.log", 
                    encoding="utf-8", 
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
logger = logging.getLogger(__name__)

class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == "RealEstateModel":
            from model.RealEstateModelTraining import RealEstateModel
            return RealEstateModel
        return super().find_class(module, name)

model = CustomUnpickler(open("./model/model.pkl", "rb")).load()

router = APIRouter()

@router.post("/api/v1/inference")
def predict(data: features.Features, user: dict = Depends(auth.get_user)):
     
	logger.info(f"Features: {data.dict()}")
	prediction_df = pd.Series(data.dict()).to_frame().T

	prediction = model.estimate(prediction_df)
	logger.info(f"Prediction: {prediction}")

	return {"price": prediction[0]}