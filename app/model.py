import tensorflow as tf
import json
import os
import numpy as np


class VideoClassifier:
    def __init__(self):
        # Load the Keras model
        model_path = os.path.join(os.path.dirname(__file__), '../model/sign_transformer.keras')
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class labels
        labels_path = os.path.join(os.path.dirname(__file__), '../model/labels.json')
        with open(labels_path, 'r') as f:
            labels_dict = json.load(f)
            self.labels = [labels_dict[str(i)] for i in range(len(labels_dict))]
    
    def predict(self, input_data):
        """
        Make prediction on processed video frames
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        predictions = self.model.predict(np.expand_dims(input_data, axis=0))
        
        # Get the predicted class and confidence
        predicted_class_idx = predictions.argmax()
        confidence = float(predictions.max())
        
        return {
            "label": self.labels[predicted_class_idx],
            "confidence": confidence
        }