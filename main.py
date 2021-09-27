# loading required libraries
import cv2 as cv
import numpy as np
import datetime
import imutils
import time

# loading video file or from webcam
#cap = cv.VideoCapture(0)
cap = cv.VideoCapture('C:\\Users\\vasu0\\Desktop\\yolo obd\KITTI dataset sequence 07 video (360p).mp4')
whT = 320
confThreshold = 0.5 # confidence of detection is above 50 percent than it is good detection
nmsThreshold = 0.2

# LOAD MODEL
# Coco Names
classesFile = "C:\\Users\\vasu0\\Desktop\\yolo obd\\coco.names"
# creating empty matrix to take class nems one by one from coco class nameset
classNames = []
# To read names one by one from classile and to write it to classnaes object
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
# import configuration file and weights file to feed dnn(from Opencv documentations)
modelConfiguration = "C:\\Users\\vasu0\\Desktop\\yolo obd\\yolov3-320.cfg"
# pre-trained weights(from yolo darknet)
modelWeights = "C:\\Users\\vasu0\\Desktop\\yolo obd\\yolov3.weights"
# Preparation of dnn network and feeding with weights and configuration
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


# creating a function to find objects by using the out puts (3 layer output )
def findObjects(outputs, img):
    hT, wT, cT = img.shape  # height ,width, channels of image
    bbox = []  # Three empty lists to hold corresponding values  of bbox , classIds, confs
    classIds = []  # class Ids
    confs = []  # Confidence
    for output in outputs:  # for loop to find highest  class probability
        for det in output:
            scores = det[5:]  # for the first 5 row values are 0
            classId = np.argmax(scores)  # in scores array takes max value and compares with threshold
            confidence = scores[classId]
            if confidence > confThreshold:

                # Identifying  the w,h,x,y from outputs
                w, h = int(det[2] * wT), int(det[3] * hT) # In detection array (from outputs )2 and 3 columns represents width and height
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                # appending obtained values to lists
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
#To find overlaping boxes ,based on confidance it will pick max confidence box (max over lap box)
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices: # Loop over indices to draw bounding boxes
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv.rectangle(img, (x, y), (x + w, y + h), (150, 0, 255), 2) # setting Initial and corner points of image
        # Putting text by using classNames and Ids to bbox
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


prev_frame_time = 0
new_frame_time = 0
while(cap.isOpened()):
#while True:
    success, img = cap.read()
    gray = img
    #img = imutils.resize(img,width=800)
    #total_frames =0
    #total_frames = total_frames + 1
    gray = cv.resize(gray, (500, 300))
    new_frame_time = time.time()

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    # # construct a blob from the image
    '''
     BLOB stands for Binary Large OBject. A blob is a data type that can store binary data. 
     This is different than most other data types used in databases, such as integers, 
     floating point numbers, characters, and strings, which store letters and numbers.
    '''
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    #The network predocts id's from 1 but in our namelist id's start from 0 so we have to subtract 1 from class id
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)
    cv.putText(img,fps,(5,30),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    # Displaying output
    cv.imshow('Image', img)
    cv.waitKey(1)
