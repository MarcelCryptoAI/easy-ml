import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict
import joblib
import os

class LSTMModel:
    def __init__(self, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
        
    def prepare_sequences(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(features) < self.sequence_length + 1:
            return None, None
        
        scaled_features = self.scaler.fit_transform(features)
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            
            future_return = (scaled_features[i, 0] - scaled_features[i-1, 0]) / scaled_features[i-1, 0]
            
            if future_return > 0.02:
                y.append(2)  # Strong buy
            elif future_return > 0.005:
                y.append(1)  # Buy
            elif future_return < -0.02:
                y.append(-2)  # Strong sell
            elif future_return < -0.005:
                y.append(-1)  # Sell
            else:
                y.append(0)  # Hold
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple):
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(5, activation='softmax')  # 5 classes: strong_sell, sell, hold, buy, strong_buy
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, features: np.ndarray) -> Dict:
        X, y = self.prepare_sequences(features)
        
        if X is None or len(X) < 50:
            return {"success": False, "error": "Insufficient data for training"}
        
        # Convert y to 0-4 range for categorical
        y_categorical = y + 2  # -2,-1,0,1,2 -> 0,1,2,3,4
        
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_categorical[:split_idx], y_categorical[split_idx:]
        
        try:
            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            
            self.is_trained = True
            
            val_loss = min(history.history['val_loss'])
            val_accuracy = max(history.history['val_accuracy'])
            
            return {
                "success": True,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epochs_trained": len(history.history['loss'])
            }
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def predict(self, features: np.ndarray) -> Dict:
        if not self.is_trained or self.model is None:
            return {"success": False, "error": "Model not trained"}
        
        if len(features) < self.sequence_length:
            return {"success": False, "error": "Insufficient data for prediction"}
        
        try:
            scaled_features = self.scaler.transform(features)
            X = scaled_features[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            prediction = self.model.predict(X, verbose=0)[0]
            predicted_class = np.argmax(prediction)
            confidence = float(prediction[predicted_class]) * 100
            
            class_mapping = {0: "sell", 1: "sell", 2: "hold", 3: "buy", 4: "buy"}
            prediction_label = class_mapping[predicted_class]
            
            return {
                "success": True,
                "prediction": prediction_label,
                "confidence": confidence,
                "probabilities": prediction.tolist()
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