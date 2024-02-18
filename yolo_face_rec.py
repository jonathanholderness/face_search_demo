import os
from ultralytics import YOLO


model = YOLO("yolov8n-face.pt").to('cpu')

source = 'input\\faces\\DSC Rome (189).jpg'
save_dir = "/output"

results = model.predict(source=source, show=True, save=False, save_crop=True )

# print(results)

# Show and save the results (NEW)
results[0].show()
filename = results[0].save()