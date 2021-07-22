import cv2
import torch
import math

#libraries for speech
from gtts import gTTS
import os

def text_to_speech(text, filename):
    language = 'es'

    myobj = gTTS(text=text, lang=language, slow=False)
  
    myobj.save(filename)

def play_audio(filename):
    os.system("afplay "+filename)
    os.remove(filename)

def object_detection():
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    #grab frame from video file
    video = cv2.VideoCapture(0)
    video = cv2.VideoCapture('dataset/source-1.mp4')
    frameCount = 0

    videoWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH ))
    videoHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    centerWidth =  int(videoWidth // 2)

    while True:
        ret,frame = video.read()

        if ret != True:
            break

        frameRate = video.get(5)
        print("frame rate {}".format(frameRate))
        print("frameCount {}".format(frameCount))

        frameCount += 1
        #image = cv2.imread('zidane.jpg')
        if frameCount % math.floor(frameRate) == 0:
            image = frame
            # Inference
            results = model(image)  # includes NMS

            labels, coordinates = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
            total_detected = len(labels)
            x_shape, y_shape = image.shape[1], image.shape[0]

            """
            print("labels: ", labels)
            print("cord: ", coordinates)
            print("model.names {}".format(model.names))
            print("objects detected {}".format(total_detected))

            print("image size: {}".format(image.shape[:2]))
            """

            for i in range(total_detected):
                row = coordinates[i]
                confidence = row[4]
                if confidence >= 0.25:
                    className = model.names[int(labels[i])]
                    if className in ["person", "car"]:
                        color = (0, 255, 0)

                        cv2.line(image, (centerWidth, 0), (centerWidth, videoHeight), color) 
                        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        className = className.replace("person", "persona").replace("car", "auto")
                        text = className + " " + str(int(confidence*100))+"%"
                        cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    #print(text)
            #results.show()  # or .show()

            cv2.imshow('video', frame)
            cv2.waitKey(0)

#filename = 'saludo.mp3'
#text_to_speech('Wena charbel culiao', filename)
#play_audio(filename)

object_detection()