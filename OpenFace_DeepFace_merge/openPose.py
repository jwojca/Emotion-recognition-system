import pandas as pd
import cv2  
import numpy as np
import math
from matplotlib.patches import Ellipse
import numpy as np
import argparse
import matplotlib.path as mpath
from matplotlib.path import Path
from matplotlib.patches import Circle

protoFile = r'C:\Users\hwojc\Desktop\Diplomka\OpenPose\repo\openpose\models\pose\mpi\pose_deploy_linevec_faster_4_stages.prototxt'
weightsFile = r'C:\Users\hwojc\Desktop\Diplomka\OpenPose\repo\openpose\models\pose\mpi\pose_iter_160000.caffemodel'
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


import matplotlib.path as mpath
import numpy as np

def customEllipse(center, ellAngle, diameter, arcStart, arcEnd):
    # Define the arc parameters
    width = int(diameter/1.5) * 2
    height = int(diameter/1.2) * 2
    if arcStart < arcEnd:
        theta1 = ellAngle + arcStart
        theta2 = ellAngle + arcEnd
    else:
        theta1 = ellAngle + arcEnd
        theta2 = ellAngle + arcStart
    
    # Compute the arc coordinates
    cx, cy = center
    rx = width / 2
    ry = height / 2
    t = np.linspace(np.deg2rad(theta1), np.deg2rad(theta2), 50)
    arcCoords = np.column_stack((cx + rx * np.cos(t), cy + ry * np.sin(t)))
    
    # Add the connecting line
    x1 = cx + rx * np.cos(np.deg2rad(theta1))
    y1 = cy + ry * np.sin(np.deg2rad(theta1))
    x2 = cx + rx * np.cos(np.deg2rad(theta2))
    y2 = cy + ry * np.sin(np.deg2rad(theta2))
    lineCoords = [[x1, y1], [x2, y2]]
    allCoords = np.vstack((arcCoords, lineCoords))
    
    # Create the path
    e = mpath.Path(allCoords)
    return e



def isInside(e, point):
    # Check if the point is inside the path
    return e.contains_point(point)

def drawCustomEllipse(e, frame, color):
    contours = e.to_polygons()
    cv2.polylines(frame, np.int32([contours]), True, color, 2)
    return frame

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
    return frame

def handsPos(frame, points):
    #handPos is array containing vals of (RHInBottFace, LHInBottFace, RHInTopFace, LHInTopFace, rhRaised, lhRaised, RHInChes, LHInChest)
    handsPos = [False, False, False, False, False, False, False, False]
    headDetected = points[0] and points[1]
    chestDetected = points[14] and points[1]
    rightWrist = points[4]
    leftWrist = points[7]
    
    if headDetected:
        diameter = math.dist(points[0], points[1])
        midPoint = [0, 0]
        # Define two points
        pointArr0 = np.array(points[0])
        pointArr1 = np.array(points[1])

        # Calculate the midpoint
        midpoint = (pointArr0 + pointArr1) / 2
        weight = 0.4
        adjMidpointTop = ((1 - weight) * pointArr1[0] + weight * pointArr0[0], (1 - weight) * pointArr1[1] + weight * pointArr0[1])
        topElCent = tuple(np.round(adjMidpointTop).astype(int))

        weight = 0.15
        adjMidpointBot = ((1 - weight) * pointArr1[0] + weight * pointArr0[0], (1 - weight) * pointArr1[1] + weight * pointArr0[1])
        botElCent = tuple(np.round(adjMidpointBot).astype(int))

        

        center = tuple(np.round(midpoint).astype(int))
        ellAngle = angleFromVertical(points[0], points[1])
        #cv2.circle(frame, center, round((diameter/2) * 1.2), (255, 0, 0), thickness = 2)

        bottEl = customEllipse(botElCent, ellAngle, diameter, 0, 180)
        topEl = customEllipse(topElCent, ellAngle, diameter, 180, 360)
        
        frame = drawCustomEllipse(bottEl, frame, (255, 0, 0))
        frame = drawCustomEllipse(topEl, frame, (0, 0, 255))
        
        
        e = Ellipse((int(center[0]), int(center[1])), 2*int(diameter/1.5), 2*int(diameter/1.2), ellAngle)
        

        if(rightWrist and not leftWrist):
            #Check if is in face area
            if isInside(bottEl, rightWrist):
                #print("Right hand in face area")
                handsPos[0] = True
            elif isInside(topEl, rightWrist):
                handsPos[2] = True

            if rHandRaised(points, handsPos[0] or handsPos[2]):
                handsPos[4] = True

            
        elif(leftWrist and not rightWrist):
            if isInside(bottEl, leftWrist):
                #print("Left hand in face area")
                handsPos[1] = True
            elif isInside(topEl, leftWrist):
                handsPos[3] = True

            if lHandRaised(points, handsPos[1] or handsPos[3]):
                handsPos[5] = True

        elif(rightWrist and leftWrist):
            #In Face
            if isInside(bottEl, rightWrist):
                handsPos[0] = True
            elif isInside(topEl, rightWrist):
                handsPos[2] = True

            if isInside(bottEl, leftWrist):
                handsPos[1] = True
            elif isInside(topEl, leftWrist):
                handsPos[3] = True

            #Raised
            if rHandRaised(points, handsPos[0] or handsPos[2]):
                handsPos[4] = True
            if lHandRaised(points, handsPos[1] or handsPos[3]):
                handsPos[5] = True
 

    if chestDetected:
        chest = points[14]
        neck = points[1]
        diameter = int(1.1 * math.dist(chest, neck)/2)
        cv2.circle(frame, chest, diameter, (255, 0, 0), thickness = 2)
        chestArea = Circle(chest, diameter)


        if rightWrist and not leftWrist:
            if chestArea.contains_point(rightWrist):
                #print("RH in chest")
                handsPos[6] = True
        elif not rightWrist and  leftWrist:
            if chestArea.contains_point(leftWrist):
                #print("LH in chest")
                handsPos[7] = True
        elif rightWrist and  leftWrist:
            if chestArea.contains_point(rightWrist):
                #print("RH in chest")
                handsPos[6] = True
            if chestArea.contains_point(leftWrist):
                #print("LH in chest")
                handsPos[7] = True




    return (frame, handsPos)
        

def rHandRaised(points, inFace):
    rightWrist = points[4]
    rightEl = points[3]
    neck = points[1]
    angle = 40.0
     #if not in face -> check if raised
    if not inFace:
        if rightEl and rightEl[1] <= neck[1]:
            return True
        elif rightEl and rightWrist[1] <= rightEl[1] and abs(angleFromVertical(rightWrist, rightEl)) < angle:
            return True
    else:
        return False

def lHandRaised(points, inFace):
    leftwrist = points[7]
    leftEl = points[6]
    neck = points[1]
    angle = 40.0
     #if not in face -> check if raised
    if not inFace:
        if leftEl and leftEl[1] <= neck[1]:
            return True
        elif leftEl and leftwrist[1] <= leftEl[1] and abs(angleFromVertical(leftwrist, leftEl)) < angle:
            return True
    else:
        return False

def GetPoints(output, frame):
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