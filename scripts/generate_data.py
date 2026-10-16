import cv2
import numpy as np
import csv
import os

# Create data directory
os.makedirs("data", exist_ok=True)

# Define gestures and their synthetic contour parameters
# We will generate synthetic shapes.
gestures = ["HELLO", "PEACE", "FIST", "OKAY"]

def generate_shape_image(gesture, width=640, height=480, noise=True):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Center and scale
    cx, cy = width // 2, height // 2
    scale = np.random.uniform(0.8, 1.2)
    
    if gesture == "HELLO":
        # 5 fingers spread out
        pts = np.array([
            [cx-40, cy+100], [cx-60, cy-50], [cx-40, cy-60], [cx-20, cy+20],
            [cx-30, cy-80], [cx-10, cy-90], [cx, cy+10],
            [cx, cy-100], [cx+20, cy-100], [cx+20, cy+10],
            [cx+30, cy-80], [cx+50, cy-70], [cx+40, cy+20],
            [cx+80, cy-20], [cx+90, cy], [cx+50, cy+80],
            [cx+30, cy+120], [cx-30, cy+120]
        ], np.int32)
    elif gesture == "PEACE":
        # 2 fingers (V shape)
        pts = np.array([
            [cx-40, cy+100], [cx-40, cy], [cx-30, cy-100], [cx-10, cy-100], [cx, cy],
            [cx+10, cy-100], [cx+30, cy-100], [cx+30, cy+20],
            [cx+50, cy+20], [cx+50, cy+80], [cx+30, cy+120], [cx-30, cy+120]
        ], np.int32)
    elif gesture == "FIST":
        # Roundish blob
        pts = np.array([
            [cx-40, cy+50], [cx-50, cy], [cx-30, cy-40], [cx+30, cy-40],
            [cx+50, cy], [cx+40, cy+50]
        ], np.int32)
    elif gesture == "OKAY":
        # A circle with 3 fingers up
        pts = np.array([
            [cx-40, cy+100], [cx-60, cy+20], [cx-20, cy-20], [cx, cy+20],
            [cx-30, cy+40], [cx-30, cy+60], [cx-10, cy+60], [cx, cy+20], # hole representation via outer contour
            [cx+10, cy-80], [cx+30, cy-80], [cx+30, cy+20],
            [cx+40, cy-70], [cx+60, cy-60], [cx+50, cy+20],
            [cx+70, cy-20], [cx+80, cy-10], [cx+50, cy+80],
            [cx+30, cy+120], [cx-30, cy+120]
        ], np.int32)
    else:
        pts = np.array([[cx-10, cy-10], [cx+10, cy-10], [cx+10, cy+10], [cx-10, cy+10]], np.int32)

    # Apply scaling and random rotation/translation
    pts = pts - np.array([cx, cy])
    pts = pts * scale
    
    angle = np.random.uniform(-15, 15)
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    pts = np.dot(pts, R.T)
    
    tx = np.random.uniform(-20, 20)
    ty = np.random.uniform(-20, 20)
    pts = pts + np.array([cx + tx, cy + ty])
    
    pts = pts.astype(np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # Draw hand
    color = (180, 215, 255) # BGR Skin color
    cv2.fillPoly(img, [pts], color)
    
    if noise:
        # Add some noise
        noise_img = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        img = cv2.add(img, noise_img)
        
    return img, pts

def extract_features(img):
    # Convert to YCrCb for skin segmentation
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 130, 80])
    upper_skin = np.array([255, 180, 135])
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Noise removal
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 1000:
        return None
        
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
                if d > 10000: # Threshold for depth
                    # calculate angle
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

    return [area, perimeter, aspect_ratio, circularity, solidity, defects_count]

def generate_dataset(num_samples_per_class=100):
    print("Generating synthetic dataset...")
    features_list = []
    labels_list = []
    
    for gesture in gestures:
        for _ in range(num_samples_per_class):
            img, _ = generate_shape_image(gesture)
            feats = extract_features(img)
            if feats is not None:
                features_list.append(feats)
                labels_list.append(gesture)
                
    # Save to CSV
    with open("data/dataset.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Area", "Perimeter", "AspectRatio", "Circularity", "Solidity", "DefectsCount", "Label"])
        for feat, label in zip(features_list, labels_list):
            writer.writerow(feat + [label])
    print(f"Dataset generated with {len(labels_list)} samples.")

def generate_sample_video():
    print("Generating sample video...")
    out = cv2.VideoWriter("data/sample_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
    
    for gesture in gestures + ["FIST"]: # FIST as unknown/rest
        for _ in range(20): # 2 seconds per gesture
            img, _ = generate_shape_image(gesture, noise=True)
            cv2.putText(img, f"True: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(img)
            
    # Add some empty/no-hand frames
    for _ in range(20):
        img = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        out.write(img)
        
    out.release()
    print("Sample video generated at data/sample_video.mp4.")

if __name__ == "__main__":
    generate_dataset()
    generate_sample_video()
