import cv2
import detector
import mediapipe as mp
from mediapipe import ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.tasks.python.vision import FaceLandmarkerResult


cap = cv2.VideoCapture(0)

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(width)
print(height)

counter = 0

modelPath = "face_landmarker.task"
baseOptions = mp.tasks.BaseOptions
faceLandmarker = mp.tasks.vision.FaceLandmarker
faceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
visionRunningMode = mp.tasks.vision.RunningMode
blendshape_result = mp.tasks.vision.FaceLandmarkerResult

options = faceLandmarkerOptions(
    base_options=baseOptions(model_asset_path=modelPath),
    running_mode=visionRunningMode.IMAGE)

landmarker = faceLandmarker.create_from_options(options)

while(True):
  # 從攝影機擷取一張影像
  ret, frame = cap.read()

  image = mp.Image(image_format=ImageFormat.SRGB, data=frame)

  face_landmarker_result = landmarker.detect(image)

  annotatedImage = detector.draw_landmarks_on_image(frame, face_landmarker_result)

  # 顯示圖片
  cv2.imshow('frame', annotatedImage)

  k = cv2.waitKey(1)

  #If space pressed 
  if k%256 == 32:
    # name = "opencv_frame_{}.bgr".format(counter)
    name = f"opencv_frame_{counter}.png"
    counter += 1
    cv2.imwrite(name, frame)

    image_data = cv2.imread(name)
    image = mp.Image(image_format=ImageFormat.SRGB, data=image_data)

    faceLandmarkerResult = detector.getResult(image)

    annotatedImage = detector.draw_landmarks_on_image(image_data, faceLandmarkerResult)
    # cv2.imwrite("annotated_image.png", annotatedImage)

  # 若按下 q 鍵則離開迴圈
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# 釋放攝影機
cap.release()

# 關閉所有 OpenCV 視窗 
cv2.destroyAllWindows()
#
# detection_result = landmarker.detect(image)
#
# detector.draw_landmarks_on_image(annotatedImage.numpy_view(), detection_result)

plotImage = plt.imshow(annotatedImage)
plt.show()