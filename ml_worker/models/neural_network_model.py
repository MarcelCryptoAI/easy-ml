import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict
import joblib
import os

class NeuralNetworkModel:
    def __init__(self):
        self.model = None
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
    
    def build_model(self, input_dim: int):
        self.model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(5, activation='softmax')  # 5 classes
        ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
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
            if self.model is None:
                self.build_model(X_scaled.shape[1])
            
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=64,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=15, 
                        restore_best_weights=True,
                        monitor='val_accuracy'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        patience=7, 
                        factor=0.5,
                        monitor='val_loss'
                    )
                ]
            )
            
            self.is_trained = True
            
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            
            return {
                "success": True,
                "test_accuracy": float(test_accuracy),
                "test_loss": float(test_loss),
                "epochs_trained": len(history.history['loss'])
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def predict(self, features: np.ndarray) -> Dict:
        if not self.is_trained or self.model is None:
            return {"success": False, "error": "Model not trained"}
        
        if len(features) == 0:
            return {"success": False, "error": "No features provided"}
        
        try:
            X = features[-1:] if len(features.shape) == 2 else features.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            prediction_proba = self.model.predict(X_scaled, verbose=0)[0]
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
    
    def save_model(self, filepath: str):
        if self.model is not None:
            self.model.save(filepath + "_model.h5")
            joblib.dump(self.scaler, filepath + "_scaler.pkl")
    
    def load_model(self, filepath: str):
        if os.path.exists(filepath + "_model.h5"):
            self.model = tf.keras.models.load_model(filepath + "_model.h5")
            self.scaler = joblib.load(filepath + "_scaler.pkl")
            self.is_trained = True
            return True
        return False