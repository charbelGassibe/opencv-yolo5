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
    model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')  # local repo

    #grab frame from video file
    #video = cv2.VideoCapture(0)
    video = cv2.VideoCapture('dataset/video-4.mp4')
    frameCount = 0

    videoWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH ))
    videoHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #centerWidth =  int(videoWidth // 2)

    #print("width {} x height {}".format(videoWidth, videoHeight))

    while True:
        ret,frame = video.read()

        if ret != True:
            break

        #frameRate = video.get(5)
        frameRate = 60
        #print("frame rate {}".format(frameRate))
        #print("frameCount {}".format(frameCount))

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
            """
            print("image size: {}".format(image.shape[:2]))

            texto_a_reproducir = ""
            contador_personas = 0
            contador_autos = 0

            for i in range(total_detected):
                row = coordinates[i]
                confidence = row[4]
                if confidence >= 0.25:
                    className = model.names[int(labels[i])]
                    if className in ["person", "car"]:
                        color = (0, 255, 0)

                        #cv2.line(frame, (centerWidth, 0), (centerWidth, videoHeight), color) 
                        x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)

                        objectSize = ( x2 - x1, y2 - y1)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        className = className.replace("person", "persona").replace("car", "auto")
                        text = className + " " + str(int(confidence*100))+"%"
                        cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        
                        widthPercentage = objectSize[0] * 100 / videoWidth
                        heightPercentage = objectSize[1] * 100 / videoHeight
                        
                        MIN_SIZE_PERCENTAGE = 0 
                        MAX_SIZE_PERCENTAGE = 0
                        if className == "persona":
                            MAX_SIZE_PERCENTAGE = 14
                            MIN_SIZE_PERCENTAGE = 5                        
                        if className == "auto":
                            MAX_SIZE_PERCENTAGE = 40
                            MIN_SIZE_PERCENTAGE = 20
                        
                        if  MIN_SIZE_PERCENTAGE >= widthPercentage <= MAX_SIZE_PERCENTAGE or MIN_SIZE_PERCENTAGE >= heightPercentage <= MAX_SIZE_PERCENTAGE:
                            center = (objectSize[0]//2 +x1, objectSize[1]//2 +y1)
                            cv2.circle(frame, center, 3, (0, 0, 255), -1)

                            #distanceFromCenter = centerWidth - center[0]
                            #distanceFromCenter = distanceFromCenter if distanceFromCenter > 0 else distanceFromCenter*-1

                            if className == "persona":
                                contador_personas += 1
                            if className == "auto":
                                contador_autos += 1

                            print("{} {}% has size {} percentage {} x {} center={} ".format(className,str(int(confidence*100)),objectSize, widthPercentage, heightPercentage, center))
                        #print("{} {}% has size {} percentage {} x {} ".format(className,str(int(confidence*100)),objectSize, widthPercentage, heightPercentage))


                    #print(text)
            #results.show()  # or .show()
            
            if contador_personas > 0:
                if contador_personas == 1:
                    texto_a_reproducir = texto_a_reproducir+ " {} persona".format("una")
                else:
                    texto_a_reproducir = texto_a_reproducir+ " {} personas".format(contador_personas)
            
            if contador_autos > 0:
                if contador_autos == 1:
                    texto_a_reproducir = texto_a_reproducir+ " {} auto".format("un")
                else:
                    texto_a_reproducir = texto_a_reproducir+ " {} autos".format(contador_autos)

            print(texto_a_reproducir)

            if texto_a_reproducir:
                filename = "alerta.mp3"
                text_to_speech(texto_a_reproducir, filename)
                play_audio(filename)


            #cv2.imshow('video', frame)
            cv2.waitKey(1)

#filename = 'saludo.mp3'
#text_to_speech('Wena charbel culiao', filename)
#play_audio(filename)

object_detection()