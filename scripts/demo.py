import cv2
import os
import json
from backend.vision import VisionPipeline
from backend.aws_integration import AWSIntegration

def run_demo():
    print("Starting Demonstration Script...")
    
    input_video_path = "data/sample_video.mp4"
    output_video_path = "data/demo_output.mp4"
    
    if not os.path.exists(input_video_path):
        print(f"Error: {input_video_path} not found.")
        return
        
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    pipeline = VisionPipeline()
    aws = AWSIntegration()
    
    predictions_log = []
    
    frame_count = 0
    print("Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, prediction, confidence = pipeline.process_frame(frame)
        
        predictions_log.append({
            "frame": frame_count,
            "prediction": prediction,
            "confidence": float(confidence)
        })
        
        out.write(processed_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames. Output saved to {output_video_path}")
    
    # Save log and attempt upload
    log_path = "data/demo_predictions.json"
    with open(log_path, "w") as f:
        json.dump(predictions_log, f, indent=4)
        
    print("Attempting to upload evaluation artifacts...")
    aws.upload_evaluation_metrics(log_path)
    print("Demonstration completed.")

if __name__ == "__main__":
    run_demo()
