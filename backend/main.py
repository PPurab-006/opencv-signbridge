from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import json
from backend.vision import VisionPipeline
from backend.aws_integration import AWSIntegration

app = FastAPI(title="SignBridge API", description="OpenCV AI Competition 2026")

# Attempt AWS init
aws_service = AWSIntegration()

# Vision Pipeline
vision_pipeline = VisionPipeline()

# Serve static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def get():
    with open("frontend/index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            # The client will send a base64 encoded image
            if data.startswith("data:image"):
                header, encoded = data.split(",", 1)
                img_data = base64.b64decode(encoded)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process frame
                processed_frame, prediction, confidence = vision_pipeline.process_frame(frame)
                
                # Encode processed frame back to base64
                _, buffer = cv2.imencode('.jpg', processed_frame)
                b64_image = base64.b64encode(buffer).decode('utf-8')
                
                # Send result back
                result = {
                    "prediction": prediction,
                    "confidence": float(confidence),
                    "image": f"data:image/jpeg;base64,{b64_image}"
                }
                await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        print("Client disconnected")

@app.post("/api/aws/upload-metrics")
async def upload_metrics():
    success = aws_service.upload_evaluation_metrics()
    return {"status": "success" if success else "failed", "fallback_used": not aws_service.enabled}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
