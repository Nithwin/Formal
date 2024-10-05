from ultralytics import YOLO
# Initialize the model
model = YOLO('./yolov8n.pt')

# Start training
results = model.train(data='C:/Users/nithwin/Desktop/idcard/id/data.yaml', epochs=50, imgsz=140)
