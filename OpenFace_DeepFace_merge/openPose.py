import pandas as pd
import cv2  
import numpy as np
import math
from matplotlib.patches import Ellipse
import numpy as np
import argparse

protoFile = r'C:\Users\hwojc\Desktop\Diplomka\OpenPose\repo\openpose\models\pose\mpi\pose_deploy_linevec_faster_4_stages.prototxt'
weightsFile = r'C:\Users\hwojc\Desktop\Diplomka\OpenPose\repo\openpose\models\pose\mpi\pose_iter_160000.caffemodel'
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

def angleFromVertical(p1, p2):
    """Calculate the angle in degrees between the line connecting points p1 and p2
    and the vertical axis."""
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    angle = math.atan2(deltaX, deltaY)
    return -math.degrees(angle)

def DrawSkeleton(frame, points):
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            
        if points[0] and points[1]:
            diameter = math.dist(points[0], points[1])
            midPoint = [0, 0]
            # Define two points
            pointArr0 = np.array(points[0])
            pointArr1 = np.array(points[1])

            # Calculate the midpoint
            midpoint = (pointArr0 + pointArr1) / 2
            center = tuple(np.round(midpoint).astype(int))
            ellAngle = angleFromVertical(points[0], points[1])
            #cv2.circle(frame, center, round((diameter/2) * 1.2), (255, 0, 0), thickness = 2)
            center2 = tuple(np.round(points[1]).astype(int))
            cv2.ellipse(frame, center, (int(diameter/1.5), int(diameter/1.2)), ellAngle, 0, 360, (255, 0, 0), thickness = 2)

            e = Ellipse((int(center[0]), int(center[1])), 2*int(diameter/1.5), 2*int(diameter/1.2), ellAngle)
            rightWrist = points[4]
            leftWrist = points[7]
            if(rightWrist and not leftWrist):
                if e.contains_point(rightWrist):
                    print("Right hand in face area")
            elif(leftWrist and not rightWrist):
                if e.contains_point(leftWrist):
                    print("Left hand in face area")
            elif(rightWrist and leftWrist):
                if e.contains_point(leftWrist) and e.contains_point(rightWrist):
                    print("Both hands in face area")
            else:
                print("No hand in face area")
    return frame

def GetPoints(output, frame, frameCopy):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1
     
    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)
    return points

def loadModel():
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    #imgPath = r'C:\Users\hwojc\Desktop\Diplomka\Repo\OpenPose\OpenPose\single.jpeg'
    """
    parser = argparse.ArgumentParser(description='Run keypoint detection')
    parser.add_argument("--device", default="gpu", help="Device to inference on")
    parser.add_argument("--image_file", default=imgPath, help="Input image")
    args = parser.parse_args()
    """
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net