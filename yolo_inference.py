from ultralytics import YOLO

model = YOLO('yolov8l')  #version 8 l- large
results = model.predict('input_video/08fd33_4.mp4', save=True)  #save as a output video
print (results[0])
print("****************************************")
for box in results[0].boxes:
    print (box)

'''
ultralytics.engine.results.Boxes object with attributes:

cls: tensor([0.]) ~ class ID...[0]referes to person
conf: tensor([0.8497]) ~ how confident the model is... more the confidence- more certain it is
data: tensor([[532.8365, 686.9611, 579.4390, 786.6328,   0.8497,   0.0000]])
id: None
is_track: False
orig_shape: (1080, 1920)
shape: torch.Size([1, 6])
~Bounding Box Config~ 
xywh: tensor([[556.1378, 736.7970,  46.6025,  99.6717]]) 
xywhn: tensor([[0.2897, 0.6822, 0.0243, 0.0923]])
xyxy: tensor([[532.8365, 686.9611, 579.4390, 786.6328]]) ~ we will use this
xyxyn: tensor([[0.2775, 0.6361, 0.3018, 0.7284]])

'''