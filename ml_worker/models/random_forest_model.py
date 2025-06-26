import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict
import joblib
import os

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_labels(self, features: np.ndarray) -> np.ndarray:
        if len(features) < 2:
            return None
        
        returns = np.diff(features[:, 0]) / features[:-1, 0]
        
        labels = []
        for ret in returns:
            if ret > 0.02:
                labels.append(2)  # Strong buy
            elif ret > 0.005:
                labels.append(1)  # Buy
            elif ret < -0.02:
                labels.append(-2)  # Strong sell
            elif ret < -0.005:
                labels.append(-1)  # Sell
            else:
                labels.append(0)  # Hold
        
        return np.array(labels)
    
    def train(self, features: np.ndarray) -> Dict:
        if len(features) < 100:
            return {"success": False, "error": "Insufficient data for training"}
        
        labels = self.prepare_labels(features)
        if labels is None:
            return {"success": False, "error": "Could not create labels"}
        
        X = features[:-1]  # Remove last row to match labels
        y = labels + 2  # Convert to 0-4 range
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        try:
            self.model.fit(X_train, y_train)
            
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            feature_importance = self.model.feature_importances_
            top_features = np.argsort(feature_importance)[-10:]
            
            return {
                "success": True,
                "accuracy": float(accuracy),
                "top_features": top_features.tolist(),
                "feature_importance": feature_importance.tolist()
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def predict(self, features: np.ndarray) -> Dict:
        if not self.is_trained:
            return {"success": False, "error": "Model not trained"}
        
        if len(features) == 0:
            return {"success": False, "error": "No features provided"}
        
        try:
            X = features[-1:] if len(features.shape) == 2 else features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prediction_proba = self.model.predict_proba(X_scaled)[0]
            predicted_class = np.argmax(prediction_proba)
            confidence = float(prediction_proba[predicted_class]) * 100
            
            class_mapping = {0: "sell", 1: "sell", 2: "hold", 3: "buy", 4: "buy"}
            prediction_label = class_mapping[predicted_class]
            
            return {
                "success": True,
                "prediction": prediction_label,
                "confidence": confidence,
                "probabilities": prediction_proba.tolist()
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_feature_importance(self) -> Dict:
        if not self.is_trained:
            return {"success": False, "error": "Model not trained"}
        
        return {
            "success": True,
            "feature_importance": self.model.feature_importances_.tolist()
        }
    
    def save_model(self, filepath: str):
        if self.is_trained:
            joblib.dump(self.model, filepath + "_model.pkl")
            joblib.dump(self.scaler, filepath + "_scaler.pkl")
    
    def load_model(self, filepath: str):
        if os.path.exists(filepath + "_model.pkl"):
            self.model = joblib.load(filepath + "_model.pkl")
            self.scaler = joblib.load(filepath + "_scaler.pkl")
            self.is_trained = True
            return True
        return False