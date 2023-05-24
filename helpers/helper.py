import math
import numpy as np
import cv2

def convertRadiantoDegree(angle):
    return angle * 180 / np.pi % 180

def convertDegreetoRadian(angle):
    return angle * np.pi / 180 % np.pi

def calculateDirection(p1, p2, inDegree=False):
    radian = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) % np.pi
    if inDegree:
        return convertRadiantoDegree(radian)
    else:
        return radian

# draw lines on the input image
# calculate the direction of the line 
# and draw the corresponding line on the label with HSV color
# while Hue value is the direction of the line
def drawLine(input, label_HSV, p1, p2, thickness=2):
    direction = calculateDirection(p1, p2, inDegree=True)
    direction = int(direction)
    cv2.line(input, p1, p2, (0, 0, 0), thickness)
    cv2.line(label_HSV, p1, p2, (direction, 255, 255), thickness)
    return input, label_HSV

def randomlyDrawLine(input, label_HSV):
    p1 = (np.random.randint(0, input.shape[1]), np.random.randint(0, input.shape[0]))
    p2 = (np.random.randint(0, input.shape[1]), np.random.randint(0, input.shape[0]))
    thickness = np.random.randint(1, 4)
    return drawLine(input, label_HSV, p1, p2, thickness)

# for a pixel in input, if the value is (255,255,255)
# the the corresponding pixel in label is set to (0,0,0)
def removeWhite(input, label_HSV):
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i][j][0] >= 128 and input[i][j][1] >= 128 and input[i][j][2] >= 128:
                label_HSV[i][j] = [0, 0, 0]
    return label_HSV

# convert label_HSV to label_xy: [x,y]
# while x and y are the verticle and horizontal components of the direction
# and if the pixel is black, then the direction is set to (0,0)
def convertHSVtoXY(label_HSV):
    label_xy = np.zeros((label_HSV.shape[0], label_HSV.shape[1], 2))
    for i in range(label_HSV.shape[0]):
        for j in range(label_HSV.shape[1]):
            if label_HSV[i][j][1] == 0 and label_HSV[i][j][2] == 0:
                label_xy[i][j] = [0, 0]
            else:
                label_xy[i][j][0] = np.cos(convertDegreetoRadian(label_HSV[i][j][0]))
                label_xy[i][j][1] = np.sin(convertDegreetoRadian(label_HSV[i][j][0]))
    return label_xy

# convert label_xy back to label_HSV
# while the direction is the arctan2 of x and y
def convertXYtoHSV(label_xy):
    label_HSV = np.zeros((label_xy.shape[0], label_xy.shape[1], 3))
    for i in range(label_xy.shape[0]):
        for j in range(label_xy.shape[1]):

            if math.pow(label_xy[i][j][0], 2) + math.pow(label_xy[i][j][1], 2) <= 0.5:
                label_HSV[i][j] = [0, 0, 0]
            else:
                label_HSV[i][j][0] = convertRadiantoDegree(np.arctan2(label_xy[i][j][1], label_xy[i][j][0]))
                label_HSV[i][j][1] = 255
                label_HSV[i][j][2] = 255
    label_HSV = label_HSV.astype(np.uint8)
    return label_HSV