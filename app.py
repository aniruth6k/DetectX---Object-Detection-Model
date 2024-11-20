from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import torch
import os
from flask_cors import CORS

app = Flask(_name_)
CORS(app)  # Enable CORS for all routes

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLOv5 model
try:
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to(device)  # Move model to GPU if available
    model.conf = 0.25  # confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

def process_frame(frame):
    try:
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create a copy of the original frame
        output_frame = frame.copy()
        
        # Perform detection
        with torch.no_grad():  # Disable gradient calculation
            results = model(frame_rgb)
        
        # Convert results to pandas dataframe
        predictions = results.pandas().xyxy[0]
        
        # Draw bounding boxes
        for idx, detection in predictions.iterrows():
            x1, y1, x2, y2 = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            conf = float(detection['confidence'])
            label = str(detection['name'])
            
            # Draw rectangle
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add background for text
            text = f'{label} {conf:.2f}'
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(output_frame, (x1, y1-30), (x1 + text_width, y1), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(output_frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
        return output_frame
    
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return
    
    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Error: Could not read frame")
                break
            
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                print("Error: Could not encode frame")
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read and process the image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Process the image
        processed_img = process_frame(img)
        
        # Ensure the upload directory exists
        upload_dir = os.path.join('static', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the processed image
        output_path = os.path.join(upload_dir, 'processed_image.jpg')
        cv2.imwrite(output_path, processed_img)
        
        return jsonify({'processed_image': '/static/uploads/processed_image.jpg',
                       'status': 'success'})
    
    except Exception as e:
        print(f"Error in detect_image: {e}")
        return jsonify({'error': str(e)}), 500

if _name_ == '_main_':
    # Create uploads directory if it doesn't exist
    os.makedirs('static/uploads', exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
