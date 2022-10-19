import onnxruntime
import cv2
import numpy as np


def preprocess(image):
    height = 256
    width = 512
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.
    image = (image - mean) / std
    image = image.astype(np.float32)
    
    tensor = image.transpose(2, 0, 1).reshape(1, 3, height, width)
    return tensor
    

session = onnxruntime.InferenceSession(
    "workspace/self_driving/model/ldrn_256x512.onnx",
    providers=["CPUExecutionProvider"]
)

image = cv2.imread("workspace/self_driving/media/test_image_01.jpg")
image = preprocess(image)

prob = session.run(["2499"], {"input.1":image})[0]
prob =  -5*prob[0,0] + 255
prob = prob[int(prob.shape[0]*0.18):]
prob = 255-prob
cv2.imwrite("output.jpg", prob)
