from datetime import datetime
import os
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

import xgboost

# Run the API:
# 1. Open a terminal
# 2. Activate env (run: conda activate store_sale_prediction_env)
# 3. Move to repo root
# 4. Run: uvicorn src.main:app --reload


FEATURE_COLUMNS = [
    "Open",
    "sqrt_Sales_lag14",
    "Sales_lag14",
    "Sales_lag1",
    "Sales_lag28",
    "sqrt_Sales_lag28",
    "Promo",
    "Sales_lag7",
]


app = FastAPI(
    title="Store Sales Prediction API",
    description="API for predicting store sales using engineered lag features",
    version="1.0.0",
)


class SalesPredictionInput(BaseModel):
    """Input schema for single prediction built from engineered lag features."""

    Open: float = Field(..., description="Store open flag (0/1)")
    sqrt_Sales_lag14: float = Field(..., description="Square root of sales 14 days ago")
    Sales_lag14: float = Field(..., description="Sales 14 days ago")
    Sales_lag1: float = Field(..., description="Sales 1 day ago")
    Sales_lag28: float = Field(..., description="Sales 28 days ago")
    sqrt_Sales_lag28: float = Field(..., description="Square root of sales 28 days ago")
    Promo: float = Field(..., description="Promotion flag or strength")
    Sales_lag7: float = Field(..., description="Sales 7 days ago")

    class Config:
        json_schema_extra = {
            "example": {
                "Open": 1.0,
                "sqrt_Sales_lag14": 32.40,
                "Sales_lag14": 1050.0,
                "Sales_lag1": 980.0,
                "Sales_lag28": 1015.0,
                "sqrt_Sales_lag28": 31.86,
                "Promo": 1.0,
                "Sales_lag7": 990.0,
            }
        }


class SalesPredictionOutput(BaseModel):
    """Output schema for prediction"""

    predicted_sales: float = Field(..., description="Predicted sales amount")
    confidence: float = Field(..., description="Prediction confidence score")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions"""

    predictions: List[SalesPredictionOutput]
    total_records: int
    processing_time: float


class ModelHandler:
    """Handles model loading and predictions."""

    def __init__(self, model_path: str | None = None):
        if model_path is None:
            # Resolve models directory relative to this file so loading works
            # no matter where the server is launched from.
            model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "models")
            )
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.encoders = {}
        self.feature_names = FEATURE_COLUMNS.copy()
        self.model_version = "1.0.0"
        self.target_scaler_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "artifacts", "y_scaler.joblib")
        )
        self.load_model()
        self.load_target_scaler()

    def load_model(self):
        """Load the trained model and preprocessing objects."""

        try:
            model_file = os.path.join(self.model_path, "prototype_xgb_regressor.joblib")
            if os.path.exists(model_file):
                model_data = joblib.load(model_file)
                if isinstance(model_data, dict):
                    self.model = model_data.get("model")
                    self.scaler = model_data.get("scaler")
                    self.encoders = model_data.get("encoders", {})
                    loaded_features = model_data.get("feature_names")
                    if loaded_features:
                        self.feature_names = loaded_features
                    self.model_version = model_data.get("version", "1.0.0")
                else:
                    # File contains the estimator directly
                    self.model = model_data
                    if hasattr(self.model, "feature_names_in_"):
                        self.feature_names = list(self.model.feature_names_in_)
                print(f"Model loaded successfully from {model_file}")
            else:
                print(f"Model file not found at {model_file}. Using dummy model.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def load_target_scaler(self):
        """Load the scaler used to revert predictions to original scale."""

        if not os.path.exists(self.target_scaler_path):
            print(f"Target scaler not found at {self.target_scaler_path}.")
            return
        try:
            self.target_scaler = joblib.load(self.target_scaler_path)
            print(f"Loaded target scaler from {self.target_scaler_path}")
        except Exception as exc:
            print(f"Error loading target scaler: {exc}")
            self.target_scaler = None

    def inverse_scale_prediction(self, value: float) -> float:
        """Invert scaling on the prediction using the stored target scaler."""

        if self.target_scaler is None:
            return value
        try:
            arr = np.array(value, dtype=float).reshape(-1, 1)
            return float(self.target_scaler.inverse_transform(arr)[0][0])
        except Exception as exc:
            print(f"Target inverse scaling failed: {exc}")
            return value

    def preprocess_input(self, data: dict) -> np.ndarray:
        """Preprocess input data for prediction."""

        df = pd.DataFrame([data])
        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features for prediction: {missing}")

        X = df[self.feature_names].astype(float).values

        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as exc:
                raise ValueError(f"Scaler transformation failed: {exc}") from exc

        return X

    def predict(self, data: dict) -> dict:
        """Make prediction for single input."""

        try:
            X = self.preprocess_input(data)

            if self.model is not None:
                prediction = float(self.model.predict(X)[0])
                confidence = 0.85
            else:
                # Fallback heuristic uses lag features to approximate current sales
                lag_values = [
                    data.get("Sales_lag1", 0.0),
                    data.get("Sales_lag7", 0.0),
                    data.get("Sales_lag14", 0.0),
                    data.get("Sales_lag28", 0.0),
                    data.get("Sales_lag365", 0.0),
                ]
                lag_values = [val for val in lag_values if val]
                base_sales = np.mean(lag_values) if lag_values else 1000.0
                promo_factor = 1.15 if data.get("Promo", 0) else 1.0
                weekday_factor = 1.05 if data.get("DayOfWeek", 0) in (4, 5) else 1.0
                prediction = float(max(0.0, base_sales * promo_factor * weekday_factor))
                confidence = 0.65

            prediction = self.inverse_scale_prediction(prediction)

            return {
                "predicted_sales": prediction,
                "confidence": float(confidence),
                "model_version": self.model_version,
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    def batch_predict(self, data_list: List[dict]) -> List[dict]:
        """Make predictions for multiple inputs."""

        predictions = []
        for data in data_list:
            try:
                predictions.append(self.predict(data))
            except Exception as e:
                print(f"Error predicting for record: {e}")
                predictions.append(
                    {
                        "predicted_sales": 0.0,
                        "confidence": 0.0,
                        "model_version": self.model_version,
                    }
                )
        return predictions


model_handler = ModelHandler()


@app.get("/")
async def root():
    return {
        "message": "Store Sales Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/batch_predict": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_handler.model is not None,
        "model_version": model_handler.model_version,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model-info")
async def model_info():
    return {
        "model_version": model_handler.model_version,
        "model_type": type(model_handler.model).__name__ if model_handler.model else "DummyModel",
        "features": model_handler.feature_names,
        "has_scaler": model_handler.scaler is not None,
        "encoders": list(model_handler.encoders.keys()),
    }


@app.post("/predict", response_model=SalesPredictionOutput)
async def predict_sales(input_data: SalesPredictionInput):
    try:
        data = input_data.model_dump()
        result = model_handler.predict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return SalesPredictionOutput(**result)


@app.post("/batch_predict", response_model=BatchPredictionOutput)
async def batch_predict_sales(input_data: List[SalesPredictionInput]):
    try:
        start_time = datetime.now()
        data_list = [item.model_dump() for item in input_data]
        predictions = model_handler.batch_predict(data_list)
        processing_time = (datetime.now() - start_time).total_seconds()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    return BatchPredictionOutput(
        predictions=[SalesPredictionOutput(**pred) for pred in predictions],
        total_records=len(predictions),
        processing_time=processing_time,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
