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


# show versions of what packages are running to make sure that the user knows if they need to update or not
def get_versions():
    print('\n[INFO] running, please wait\n')
    print('\n[INFO] Versions installed:')
    print('[INFO] Getting Versions...\n')

    print('[INFO] cv2 version:', cv2.__version__)
    print('[INFO] argparse version:', argparse.__version__)
    print('[INFO] numpy version:', np.__version__)

    # always the last one because it is the new line one
    print('[INFO] imutils version:', imutils.__version__, '\n')


# get user input for how big they want the frame to be
def get_user_input():
    print("1. Input how big you want the frame. Ex 420x420, 720x720, 1080x1080, 1440x1440")
    print("2. Press 'r' to run the program or 'q' to exit\n")

    frame_height = int(input("height: "))
    frame_width = int(input("width: "))
    return frame_height, frame_width


# The model is trained on the 21 classes below, in order to filter out...
# unneeded information we specify which classes we want to detect

def get_model_classes():
    return ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor", "apple"]


# selection process
def select_only_what_we_want():
    return {"person": 15, "cat": 8, "dog": 12}


# make it so that the bounding boxes are random colors within the RGB spectrum
def show_colors(colors):
    return np.random.uniform(0, 255, size=(len(colors), 3))


# check if the model files are present if they are not show the error message
def see_if_files_exist(protxtfile, model_file):
    if not os.path.exists(protxtfile):
        print("[ERROR!] No protxtfile detected please check file location")
        if not os.path.exists(model_file):
            print("[ERROR!] No model file detected check file location")
            return False
        else:
            return False
    return True


# get the video and put it through the model
def get_video(protxtfile, model_file):
    return cv2.dnn.readNetFromCaffe(protxtfile, model_file)


# start camera
def run_camera():
    # allow the camera sensor to warm up,
    # start the FPS counter
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    return vs, fps


def Processing(vs, fps, frame_width, frame_height, net, colors, classes):
    count = {"person": 0, "dog": 0, "cat": 0}

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
                class_confidence = int(detections[0, 0, i, 1])
                filtered_classes = select_only_what_we_want()

                # PUT CLASSES HERE
                if class_confidence in filtered_classes.values():
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # count how many dogs humans and cats we see
                    count[classes[class_confidence]] += 1

                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(classes[class_confidence],
                                                 confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  colors[class_confidence].tolist(), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_confidence].tolist(), 2)
                else:
                    if confidence > 1:
                        print("NOTHING DETECTED")

        # show the output frame
        cv2.imshow("Video Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # update the FPS counter
        fps.update()

        # check for keyboard input to break the loop
        if key == ord("q"):
            break

    # stop the timer and display FPS information
    fps.stop()
    print("number of objects:", count)
    print("[INFO] elapsed time: {:.2f} seconds".format(fps.elapsed()))
    print("[INFO] approx  FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def display():
    get_versions()

    frame_height, frame_width = get_user_input()

    while True:
        if keyboard.is_pressed('r'):
            print("\nStarting video...")

            protxtfile = "MobileNetSSD_deploy.prototxt.txt"
            model_file = "MobileNetSSD_deploy.caffemodel"

            if see_if_files_exist(protxtfile, model_file):
                net = get_video(protxtfile, model_file)
                classes = get_model_classes()
                colors = show_colors(classes)
                vs, fps = run_camera()
                Processing(vs, fps, frame_width, frame_height, net, colors, classes)

        elif keyboard.is_pressed('q'):
            print('\n[INFO] Program Terminated')
            break


if __name__ == "__main__":
    display()


