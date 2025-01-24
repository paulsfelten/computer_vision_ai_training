#For reference: https://docs.ultralytics.com/modes/train/

#From Terminal:
# cd to this directory: cd ./bot_party_training
# run this command: python3 /Users/paul.felten/bot_party_training/training_script.py

from ultralytics import YOLO

# Load a model
model = YOLO("./datasets/yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model with MPS
results = model.train(data="./data.yaml", epochs=100, imgsz=1024, cache=True)

# 53 images imgsz@1024 epochs@100 took ~2hrs; ultralytics@8.3.58; torch@2.4.0; Python@3.11.8; Apple M2;