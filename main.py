#!/usr/bin/env python3
'''
    Rio Hondo College
    CIT 128: Python Programming II
    Student Directed Project
'''

# Jared Mendoza, python-II Professor Harlow 2-16-2023
# class project


import os
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import keyboard

print("1. Input how big you want the frame. Ex 420x420, 720x720, 1080x1080, 1440x1440")
print("2. Press 'r' to run the program or 'q' to exit\n")

frame_height = int(input("height: "))
frame_width = int(input("width: "))

while True:
    if keyboard.is_pressed('r'):
        print('\n[INFO] running, please wait\n')
        print('\n[INFO] Versions installed:')
        print('[INFO] Getting Versions...\n')

        print('[INFO] cv2 version:', cv2.__version__)
        print('[INFO] argparse version:', argparse.__version__)
        print('[INFO] numpy version:', np.__version__)

        # always the last one because it is the new line one
        print('[INFO] imutils version:', imutils.__version__, '\n')

        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor", "apple"]

        protxtfile = 'MobileNetSSD_deploy.prototxt.txt'
        model_file = 'MobileNetSSD_deploy.caffemodel'

        if not os.path.exists(protxtfile):
            print("No protxtfile detected please check file location")
            if not os.path.exists(model_file):
                print("No model file detected check file location")
        else:
            net = cv2.dnn.readNetFromCaffe(protxtfile, model_file)

            COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

            # load our serialized model from disk
            print("[INFO] loading model and video stream...\n")
            net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

            # initialize the video stream, allow the camera sensor to warmup,
            # and initialize the FPS counter
            vs = VideoStream(src=0).start()
            time.sleep(2.0)
            fps = FPS().start()

            # loop over the frames from the video stream
            while True:
                # grab the frame from the threaded video stream and resize it
                frame = vs.read()
                frame = imutils.resize(frame, width=frame_width, height=frame_height)

                # grab the frame dimensions and convert it to a blob
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                             0.007843, (300, 300), 127.5)

                # pass the blob through the network and obtain the detections and
                # predictions
                net.setInput(blob)
                detections = net.forward()

                # loop over the detections
                for i in np.arange(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]

                    # filter out weak detections by ensuring the `confidence` is
                    # greater than the minimum confidence
                    if confidence > 0.2:
                        # extract the index of the class label from the
                        # `detections`
                        label_confidence = int(detections[0, 0, i, 1])

                        # Check if the detected object is an apple
                        if CLASSES[label_confidence] == "apple":
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # draw the prediction on the frame
                            label = "{}: {:.2f}%".format(CLASSES[label_confidence],
                                                         confidence * 80)
                            cv2.rectangle(frame, (startX, startY), (endX, endY),
                                          COLORS[label_confidence], 2)
                            y = startY - 15 if startY - 15 > 15 else startY + 15
                            cv2.putText(frame, label, (startX, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[label_confidence], 2)
                        else:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # draw the "This is not an apple" message on the frame
                        label = "This is not an apple"
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                                      (0, 255, 255), 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


                # show the output frame
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Program Terminated")
                    break

                # update the FPS counter
                fps.update()

            # stop the timer and display FPS information
            fps.stop()
            print("[INFO] elapsed time: {:.2f} seconds".format(fps.elapsed()))
            print("[INFO] approx  FPS: {:.2f}".format(fps.fps()))

            # do a bit of cleanup
            cv2.destroyAllWindows()
            vs.stop()
            break

    elif keyboard.is_pressed('q'):
        print('\n[INFO] Program Terminated')
        break
