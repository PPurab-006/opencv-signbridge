const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const output = document.getElementById('output');
const startBtn = document.getElementById('start-camera');
const stopBtn = document.getElementById('stop-camera');
const clearBtn = document.getElementById('clear-transcript');
const predictionBox = document.getElementById('prediction-box');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const transcript = document.getElementById('transcript');

let stream = null;
let ws = null;
let captureInterval = null;
let lastRecognized = null;
let lastRecognizedTime = 0;

startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        video.srcObject = stream;
        video.play();
        
        startBtn.disabled = true;
        stopBtn.disabled = false;

        connectWebSocket();
    } catch (err) {
        alert("Could not access camera. Please ensure permissions are granted.");
        console.error(err);
    }
});

stopBtn.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    if (captureInterval) {
        clearInterval(captureInterval);
    }
    if (ws) {
        ws.close();
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
    predictionBox.textContent = "Camera stopped.";
    predictionBox.className = "";
    output.src = "";
});

clearBtn.addEventListener('click', () => {
    transcript.innerHTML = '';
});

function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

    ws.onopen = () => {
        console.log("WebSocket connected");
        canvas.width = 640;
        canvas.height = 480;
        
        captureInterval = setInterval(() => {
            if (video.readyState === video.HAVE_ENOUGH_DATA) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const imageData = canvas.toDataURL('image/jpeg', 0.8);
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(imageData);
                }
            }
        }, 100); // 10 FPS
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // Update image
        if (data.image) {
            output.src = data.image;
        }

        // Update prediction UI
        const confPct = Math.round(data.confidence * 100);
        confidenceBar.style.width = `${confPct}%`;
        confidenceText.textContent = `Confidence: ${confPct}%`;

        if (data.prediction !== "UNKNOWN" && data.prediction !== "REST/UNKNOWN" && data.prediction !== "UNCERTAIN") {
            predictionBox.textContent = data.prediction;
            predictionBox.className = "recognized";
            
            // Add to transcript if it's a new gesture or 2 seconds have passed
            const now = Date.now();
            if (data.prediction !== lastRecognized || (now - lastRecognizedTime > 2000)) {
                addToTranscript(data.prediction, confPct);
                lastRecognized = data.prediction;
                lastRecognizedTime = now;
            }
        } else {
            predictionBox.textContent = data.prediction;
            predictionBox.className = "uncertain";
            lastRecognized = null; // Reset
        }
    };

    ws.onclose = () => {
        console.log("WebSocket disconnected");
        if (stopBtn.disabled === false) {
            // Reconnect if not manually stopped
            setTimeout(connectWebSocket, 1000);
        }
    };
}

function addToTranscript(text, confidence) {
    const li = document.createElement('li');
    const time = new Date().toLocaleTimeString();
    li.innerHTML = `<strong>[${time}]</strong> ${text} <span style="color: #6c757d; font-size: 0.8em;">(${confidence}% conf)</span>`;
    transcript.appendChild(li);
    transcript.scrollTop = transcript.scrollHeight;
}
