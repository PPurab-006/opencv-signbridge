import cv2
import numpy as np
import joblib
import os
import collections

# Load model
MODEL_PATH = "data/model.pkl"
clf = None
if os.path.exists(MODEL_PATH):
    clf = joblib.load(MODEL_PATH)

def extract_features(img):
    # Same as in generate_data.py
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 130, 80])
    upper_skin = np.array([255, 180, 135])
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask, None
        
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 1000:
        return None, mask, None
        
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    hull_indices = cv2.convexHull(c, returnPoints=False)
    defects_count = 0
    try:
        defects = cv2.convexityDefects(c, hull_indices)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 10000:
                    start = tuple(c[s][0])
                    end = tuple(c[e][0])
                    far = tuple(c[f][0])
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c_side = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = np.arccos((b**2 + c_side**2 - a**2) / (2 * b * c_side)) * 57
                    if angle <= 90:
                        defects_count += 1
    except Exception:
        pass

    return [area, perimeter, aspect_ratio, circularity, solidity, defects_count], mask, c

class VisionPipeline:
    def __init__(self):
        self.history = collections.deque(maxlen=5) # For temporal smoothing
        self.clf = clf
        
    def process_frame(self, frame):
        if frame is None:
            return frame, "Error: No frame", 0.0

        features, mask, contour = extract_features(frame)
        
        prediction = "UNKNOWN"
        confidence = 0.0
        
        if features is not None and self.clf is not None:
            # Reshape for prediction
            if hasattr(self.clf, "feature_names_in_"):
                import pandas as pd
                features_input = pd.DataFrame([features], columns=self.clf.feature_names_in_)
            else:
                features_input = np.array(features).reshape(1, -1)
            probs = self.clf.predict_proba(features_input)[0]
            max_idx = np.argmax(probs)
            confidence = probs[max_idx]
            
            if confidence > 0.6:
                pred = self.clf.classes_[max_idx]
                if pred == "FIST":
                    prediction = "REST/UNKNOWN"
                else:
                    prediction = pred
            else:
                prediction = "UNCERTAIN"
                
        # Smoothing
        self.history.append(prediction)
        
        # Most common prediction in history
        smoothed_prediction = max(set(self.history), key=self.history.count)
        
        # Draw on frame
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)
            
        cv2.putText(frame, f"Gest: {smoothed_prediction} ({confidence:.2f})", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
        return frame, smoothed_prediction, confidence
