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
import imutils
import time
import cv2
import keyboard


# get user input for how big they want the frame to be
def get_user_input():
    """Get user input for frame dimensions and return as a tuple.

    Inputs:
       int: How big the user wants to make the frame size.

    Returns:
        frame: The size will depend on user preferences.

    ValueError:
        Returns Invalid choice, defaulting to 720 x 720 for both height and width parameters


    """

    print("1. Input how big you want the frame. Ex 420x420, 720x720, 1080x1080, 1440x1440")
    print("2. Press 'ENTER' twice to run the program or 'q' to exit\n")

    try:
        frame_height = int(input("Height: "))
        if frame_height is str or None:
            raise ValueError
    except ValueError:
        print("Invalid choice, defaulting to 720 x 720")
        frame_height = 720
    try:
        frame_width = int(input("Width: "))
        if frame_width is str or None:
            raise ValueError
    except ValueError:
        print("Invalid choice, defaulting to 720 x 720")
        frame_width = 720

    return frame_height, frame_width


# The model is trained on the 21 classes below, in order to filter out...
# unneeded information we specify which classes we want to detect

def get_model_classes():
    """Return the list of classes that the model is trained on,
    the model must have all 21 classes present or else it cannot compare detected objects to all...
    ..other classes making it so that it is unable to process the detected object.

    Returns:
        text: what object is being detected in the frame.

    """

    return ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor", "apple"]


# selection process
def select_only_what_we_want():
    """Returns a dictionary of the classes selected for object detection along with their built-in integer."""

    return {"person": 15, "cat": 8, "dog": 12}


# make it so that the bounding boxes are random colors within the RGB spectrum
def show_colors(colors):
    """Return random colors within the RGB spectrum for bounding boxes.
    Param colors:
            returns a random color for when an object is detected this color...
    ..is applied to the bounding box.

    """

    return np.random.uniform(0, 255, size=(len(colors), 3))


# check if the model files are present if they are not show the error message
def see_if_files_exist(protxtfile, model_file):
    """Check if the model files exist and return a boolean.
      protxtfile:
            returns: the protxtfile that contains the images used for training.

    param model_file:
            returns: The CAFFE model file that turns the model into plain text.

    """

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
    """Load the video and return it after putting it through the model.

    protxtfile:
            returns: the protxtfile that contains the images used for training.

    param model_file:
            returns: The CAFFE model file that turns the model into plain text.
    """

    return cv2.dnn.readNetFromCaffe(protxtfile, model_file)


# start camera
def run_camera():
    """Initialize the camera and start the FPS counter.
    Takes the video-feed from the users webcam and starts the timer for the fps.

    """

    # allow the camera sensor to warm up,
    # start the FPS counter
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()
    return vs, fps


def Processing(vs, fps, frame_width, frame_height, net, colors, classes, model_confidence):
    """Process the video stream, display object detection, and count detected objects.
    param: VS
        Video-Stream takes the uses webcam and starts recording


    Param: Fps
        Frames per second measures how many frames per second the program is running at
        having the fps being known is important as proper streaming should take place
        above 30 frames per second.

    Param input:
            Returns:
                frame_width:
                    Frame_width is an input the user interacts with at the beginning that sets the width of the frame

    Param input:
            Returns:
                frame_height:
                    Frame_height is an input the user interacts with at the beginning that sets the height of the frame

    Param: net
        Using the network parameter we can now send our images for processing without creating our own Convolutional
    Neural Network instead we can now use the built-in processing from numpy.

     Param: colors
        returns:
            a random color for when an object is detected this color...
        ...is applied to the bounding box. Serves an Aesthetic purpose.

    Param: classes
        Returns:
            text: what object is being detected in the frame.

    Param: model_confidence
        input:
            parameter:
                Model_confidence is what float between 0.1 and 0.9 does the user want to set the confidence of the...
                program to, the models set confidence will effect how accuracy the objects in the frame are being assigned..
                ..to the mentioned classes.

    """

    count = {"person": 0, "dog": 0, "cat": 0}
    detected_objects = {}
    frame_index = 0
    start_time = time.time()

    while True:

        # grab the frame from the threaded video stream
        frame = vs.read()
        time.sleep(0.01)

        if frame is None:
            continue

        frame_width, frame_height = int(frame_width), int(frame_height)

        # start a timer for the fps
        elapsed_time = time.time() - start_time
        fps_amount = frame_index / elapsed_time if elapsed_time > 0 else 0
        # display FPS on the screen
        fps_info = "FPS: {:.2f}".format(fps_amount)
        cv2.putText(frame, fps_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        frame_index += 1

        # Make sure dimensions are valid before resizing
        if frame_width > 0 and frame_height > 0:
            frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        else:
            print("[FRAME ERROR!] Please provide valid frame dimensions.")
            break

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]

        # define the blob parameters more clearly
        resized_frame = cv2.resize(frame, (300, 300))
        scalefactor = 0.007843
        size = (300, 300)
        mean = 127.5
        blob = cv2.dnn.blobFromImage(resized_frame, scalefactor, size, mean)
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

            if confidence > model_confidence:
                # extract the index of the class label from the
    
                class_confidence = int(detections[0, 0, i, 1])
                filtered_classes = select_only_what_we_want()

                # PUT CLASSES HERE
                if class_confidence in filtered_classes.values():
                    box = detections[0, 0, i, 3:9] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # draw the prediction on the frame
                    label = "{}: {:.2f}%".format(classes[class_confidence],
                                                 confidence * 100)
                    # counting the objects
                    # FIXED!!!
                    class_name = classes[class_confidence]

                    # Check if the class is among the classes we're interested in
                    if class_name in filtered_classes:

                        # Define a unique key for each detected object
                        object_key = f"{class_name}_{startX}_{startY}"

                        # Get the current time
                        current_time = time.time()

                        # Check if the object is already in the detected_objects dictionary
                        if object_key not in detected_objects:
                            # If it's not, add it
                            detected_objects[object_key] = current_time
                        else:
                            # If it is, check how much time has passed since it was first detected
                            time_since_detection = current_time - detected_objects[object_key]

                            # If more than 4.5 seconds have passed, add to the count 
                            if time_since_detection >= 4.5:
                                count[class_name] += 1

                                # Update the time of detection for this object
                                detected_objects[object_key] = current_time

                    # set color
                    rectangle_color = colors[class_confidence].tolist()
                    # Draw the rectangle on the frame
                    cv2.rectangle(frame, (startX, startY), (endX, endY), rectangle_color, 2)

                    # Define the position for the label
                    y_position = startY - 15 if startY - 15 > 15 else startY + 15
                    label_position = (startX, y_position)

                    # Define the font for the label
                    label_font = cv2.FONT_HERSHEY_SIMPLEX
                    label_scale = 0.5
                    label_color = rectangle_color
                    label_thickness = 2

                    # Put the label on the frame
                    cv2.putText(frame, label, label_position, label_font, label_scale, label_color, label_thickness)

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
    print("\nObjects detected:", f"person: {count['person']}, dog: {count['dog']}, cat: {count['cat']}")

    print("[INFO] elapsed time: {:.2f} seconds".format(fps.elapsed()))
    time_passed = fps.elapsed()
    if time_passed > 0:

        print("[INFO] FPS: {:.2f}".format(fps.fps()))
    else:
        print("[INFO] waiting for time to start ")

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()


def display(model_confidence):
    """Display package versions, user input, and run object detection on the video stream.

        model_confidence: object
            Displays what settings user has chosen.

    """

    settings()

    frame_height, frame_width, = get_user_input()

    while True:
        if keyboard.is_pressed('enter'):
            print("\nStarting video...")

            protxtfile = "MobileNetSSD_deploy.prototxt.txt"
            model_file = "MobileNetSSD_deploy.caffemodel"

            if see_if_files_exist(protxtfile, model_file):
                net = get_video(protxtfile, model_file)
                classes = get_model_classes()
                colors = show_colors(classes)
                vs, fps = run_camera()
                Processing(vs, fps, frame_width, frame_height, net, colors, classes, model_confidence)

        elif keyboard.is_pressed('q'):
            print('\n[INFO] Program Terminated')
            break


if __name__ == "__main__":

    # show versions of what packages are running to make sure that the user knows if they need to update or not
    def settings():
        """sets the users preferences.

        Returns:
            text: Shows the current versions of the imports OpenCV, NumPy, and Imutils.

        float(input):
            sets: model accuracy

        ValueError:
            returns: invalid choice and sets the default confidence to 0.7

        """


    print('\n[INFO] running, please wait\n')
    print('\n[INFO] Versions installed:')
    print('[INFO] Getting Versions...\n')

    print('[INFO] cv2 version:', cv2.__version__)
    print('[INFO] numpy version:', np.__version__)

    # always the last one because it is the new line one
    print('[INFO] imutils version:', imutils.__version__, '\n')

    print("settings \n")
    try:
        model_confidence = float(input("Choose model accuracy (0.1 to 0.9): "))
        if model_confidence < 0.1 or model_confidence > 0.9:
            raise ValueError
    except ValueError:
        print("Invalid choice, defaulting to value of 0.7.")
        model_confidence = 0.7

    display(model_confidence)
