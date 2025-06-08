import cv2
import numpy as np

## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)

    if len(labels) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(labels[d][c]) * 13 + 27, 30 + eachImgHeight * d),
                              (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, labels[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver

## REORDER POINTS TO CONSISTENT FORMAT
def reorder(myPoints):
    myPoints = np.array(myPoints, dtype=np.int32).reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    diff = np.diff(myPoints, axis=1)
    myPointsNew[0] = myPoints[np.argmin(add)]     # Top-left
    myPointsNew[3] = myPoints[np.argmax(add)]     # Bottom-right
    myPointsNew[1] = myPoints[np.argmin(diff)]    # Top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]    # Bottom-left
    return myPointsNew

## WARP IMAGE BASED ON PROVIDED CORNERS
def getWarp(img, points, width, height):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width, height))
    return imgWarp

## SPLIT THE BOXES
def splitBoxes(img):
    rows = np.vsplit(img, 5)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes

## DRAW GRID ON IMAGE
def drawGrid(img, questions=5, choices=5):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)
    for i in range(0, questions + 1):
        pt1 = (0, secH * i)
        pt2 = (img.shape[1], secH * i)
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
    for i in range(0, choices + 1):
        pt1 = (secW * i, 0)
        pt2 = (secW * i, img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0), 2)
    return img

## SHOW THE SELECTED AND CORRECT ANSWERS
def showAnswers(img, myIndex, grading, ans, questions=5, choices=5):
    secW = int(img.shape[1] / choices)
    secH = int(img.shape[0] / questions)

    for x in range(0, questions):
        myAns = myIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        if grading[x] == 1:
            myColor = (0, 255, 0)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
        else:
            myColor = (0, 0, 255)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
            correctAns = ans[x]
            cv2.circle(img, ((correctAns * secW) + secW // 2, (x * secH) + secH // 2),
                       20, (0, 255, 0), cv2.FILLED)
    return img
