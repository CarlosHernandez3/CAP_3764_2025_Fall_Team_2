from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Run the API:
# 1. Open a terminal
# 2. Activate env (run in terminal: conda activate store_sale_prediction_env)
# 3. Set your working directory to the ROOT folder (run in terminal: cd your_path/CAP_3764_2025_Fall_Team_2)
# 4. Execute API (run in terminal: uvicorn src.main:app --reload)

app = FastAPI(
    title="Store Sales Prediction API",
    description="API for predicting store sales using machine learning models",
    version="1.0.0"
)

# -----------------------------
# Pydantic Models
# -----------------------------

class SalesPredictionInput(BaseModel):
    """Input schema for single prediction"""
    store_nbr: int = Field(..., ge=1, description="Store number")
    family: str = Field(..., description="Product family")
    onpromotion: int = Field(..., ge=0, description="Number of items on promotion")
    store_type: str = Field(..., description="Type of store (A, B, C, D, E)")
    cluster: int = Field(..., ge=1, description="Store cluster")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend indicator (0 or 1)")
    is_holiday: int = Field(..., ge=0, le=1, description="Holiday indicator (0 or 1)")
    oil_price: float = Field(..., ge=0, description="Oil price in USD")
    
    class Config:
        json_schema_extra = {
            "example": {
                "store_nbr": 1,
                "family": "GROCERY I",
                "onpromotion": 10,
                "store_type": "A",
                "cluster": 1,
                "date": "2024-01-15",
                "day_of_week": 0,
                "is_weekend": 0,
                "is_holiday": 0,
                "oil_price": 50.5
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


# -----------------------------
# Model Loading
# -----------------------------

class ModelHandler:
    """Handles model loading and predictions"""
    
    def __init__(self, model_path: str = "../models"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = []
        self.model_version = "1.0.0"
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            # Try to load the main model
            model_file = os.path.join(self.model_path, "prototype_xgb_regressor.joblib")
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.encoders = model_data.get('encoders', {})
                    self.feature_names = model_data.get('feature_names', [])
                    self.model_version = model_data.get('version', '1.0.0')
                print(f"Model loaded successfully from {model_file}")
            else:
                print(f"Model file not found at {model_file}. Using dummy model.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_input(self, data: dict) -> np.ndarray:
        """Preprocess input data for prediction"""
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Parse date features
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        # Encode categorical variables
        categorical_cols = ['family', 'store_type']
        for col in categorical_cols:
            if col in self.encoders:
                try:
                    df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
                except:
                    # If unseen category, use a default value
                    df[f'{col}_encoded'] = 0
            else:
                # Simple label encoding if encoder not available
                df[f'{col}_encoded'] = pd.factorize(df[col])[0]
        
        # Select features for prediction
        feature_cols = [
            'store_nbr', 'onpromotion', 'cluster', 'day_of_week',
            'is_weekend', 'is_holiday', 'oil_price', 'year', 'month', 'day',
            'family_encoded', 'store_type_encoded'
        ]
        
        # Use only available features
        available_features = [col for col in feature_cols if col in df.columns]
        X = df[available_features].values
        
        # Scale features if scaler is available
        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except:
                pass
        
        return X
    
    def predict(self, data: dict) -> dict:
        """Make prediction for single input"""
        try:
            X = self.preprocess_input(data)
            
            if self.model is not None:
                # Use actual model
                prediction = self.model.predict(X)[0]
                
                # Calculate confidence (mock - adjust based on your model)
                # For regression, we can use prediction stability or model score
                confidence = 0.85  # Placeholder
                
                # Ensure non-negative sales
                prediction = max(0, prediction)
            else:
                # Dummy prediction based on input features
                base_sales = 1000
                promotion_factor = 1 + (data['onpromotion'] * 0.05)
                weekend_factor = 1.2 if data['is_weekend'] else 1.0
                holiday_factor = 1.5 if data['is_holiday'] else 1.0
                
                prediction = base_sales * promotion_factor * weekend_factor * holiday_factor
                confidence = 0.75
            
            return {
                "predicted_sales": float(prediction),
                "confidence": float(confidence),
                "model_version": self.model_version
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def batch_predict(self, data_list: List[dict]) -> List[dict]:
        """Make predictions for multiple inputs"""
        predictions = []
        for data in data_list:
            try:
                pred = self.predict(data)
                predictions.append(pred)
            except Exception as e:
                # Skip failed predictions but continue processing
                print(f"Error predicting for record: {e}")
                predictions.append({
                    "predicted_sales": 0.0,
                    "confidence": 0.0,
                    "model_version": self.model_version
                })
        return predictions


# Initialize model handler
model_handler = ModelHandler()


# -----------------------------
# API Endpoints
# -----------------------------

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Store Sales Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/batch_predict": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/model-info": "GET - Model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_handler.model is not None,
        "model_version": model_handler.model_version,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    return {
        "model_version": model_handler.model_version,
        "model_type": type(model_handler.model).__name__ if model_handler.model else "DummyModel",
        "features": model_handler.feature_names,
        "has_scaler": model_handler.scaler is not None,
        "encoders": list(model_handler.encoders.keys())
    }


@app.post("/predict", response_model=SalesPredictionOutput)
async def predict_sales(input_data: SalesPredictionInput):
    """
    Predict sales for a single store/product combination
    """
    try:
        # Convert input to dict
        data = input_data.model_dump()
        
        # Make prediction
        result = model_handler.predict(data)
        
        return SalesPredictionOutput(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionOutput)
async def batch_predict_sales(input_data: List[SalesPredictionInput]):
    """
    Predict sales for multiple store/product combinations
    """
    try:
        start_time = datetime.now()
        
        # Convert inputs to list of dicts
        data_list = [item.model_dump() for item in input_data]
        
        # Make batch predictions
        predictions = model_handler.batch_predict(data_list)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchPredictionOutput(
            predictions=[SalesPredictionOutput(**pred) for pred in predictions],
            total_records=len(predictions),
            processing_time=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# -----------------------------
# Run with: uvicorn main:app --reload
# -----------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)