import cv2
import numpy as np
import utility

pathImage = "1.jpg"
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]  # Your answer key

count = 0

while True:
    img = cv2.imread(pathImage)
    img = cv2.resize(img, (widthImg, heightImg))
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)


    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 30, 100)   # slightly higher thresholds than (10,70)

    # (Optional) Dilate to strengthen edges
    kernel = np.ones((5, 5), np.uint8)
    imgCanny = cv2.dilate(imgCanny, kernel, iterations=1)

    try:
        imgContours = img.copy()
        imgBigContour = img.copy()

        contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

        rectCon = utility.rectContour(contours)  # returns a list of 4‐corner contours
        if len(rectCon) < 2:
            raise Exception("Less than two rectangular contours found.")

        # Sort by area, descending → largest first, then next largest
        rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
        answerContour = rectCon[0]
        gradeContour  = rectCon[1]


        biggestPoints = utility.getCornerPoints(answerContour)
        biggestPoints = utility.reorder(biggestPoints)  # order: [top-left, top-right, bottom-left, bottom-right]
        pts1 = np.float32(biggestPoints)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        gradePoints = utility.getCornerPoints(gradeContour)
        gradePoints = utility.reorder(gradePoints)
        ptsG1 = np.float32(gradePoints)
        # We know the grade‐box size is 325×150 in the warped coordinate system:
        ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))


        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

        # You can choose simple threshold or adaptive. We'll keep simple here:
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]
        # (If lighting is uneven, swap the above for adaptiveThreshold:
        # imgThresh = cv2.adaptiveThreshold(imgWarpGray, 255,
        #                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                   cv2.THRESH_BINARY_INV, 11, 2)
        # )

        boxes = utility.splitBoxes(imgThresh)
        myPixelVal = np.zeros((questions, choices))

        for i in range(questions * choices):
            row = i // choices
            col = i % choices
            myPixelVal[row][col] = cv2.countNonZero(boxes[i])

        myIndex = []
        grading = []
        threshold = 0.5 * np.max(myPixelVal)

        for x in range(questions):
            row = myPixelVal[x]
            marked = np.where(row >= threshold)[0]  # indices of choices with enough black pixels

            if len(marked) == 1:
                myIndex.append(int(marked[0]))
                grading.append(1 if ans[x] == marked[0] else 0)
            else:
                myIndex.append(-1)    # none or multiple marked
                grading.append(0)

        score = (sum(grading) / questions) * 100

        utility.showAnswers(imgWarpColored, myIndex, grading, ans)
        utility.drawGrid(imgWarpColored)

        # Create a blank drawing for bubbles, then warp it back:
        imgRawDrawings = np.zeros_like(imgWarpColored)
        utility.showAnswers(imgRawDrawings, myIndex, grading, ans)

        invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
        imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

        imgRawGrade = np.zeros((150, 325, 3), np.uint8)

        # Draw the text. Use fontScale=2 so it fits inside 150px height:
        cv2.putText(
            imgRawGrade,
            f"{int(score)}%",
            (80, 100),                     
            cv2.FONT_HERSHEY_COMPLEX,
            3,                              
            (0, 0, 255),                    
            3                               
        )

        # Warp the small grade box drawing back onto the original image:
        invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

        # Show the final result
        cv2.imshow("Final Result", imgFinal)

        imageArray = (
            [img,       imgGray,  imgCanny,      imgContours],
            [imgBigContour, imgThresh, imgWarpColored, imgFinal]
        )
        labels = [
            ["Original", "Gray", "Edges", "Contours"],
            ["Biggest Contour", "Threshold", "Warped", "Final"]
        ]
        stackedImage = utility.stackImages(imageArray, 0.5, labels)
        cv2.imshow("Result (Intermediate Steps)", stackedImage)

    except Exception as e:
        # If something goes wrong, just show blank placeholders
        print("Error:", e)
        imageArray = (
            [img, imgGray, imgCanny, imgBlank],
            [imgBlank, imgBlank, imgBlank, imgBlank]
        )
        labels = [
            ["Original", "Gray", "Edges", "Contours"],
            ["Biggest Contour", "Threshold", "Warped", "Final"]
        ]
        stackedImage = utility.stackImages(imageArray, 0.5, labels)
        cv2.imshow("Result (Intermediate Steps)", stackedImage)

    # Save on pressing 's'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(f"Scanned/myImage{count}.jpg", imgFinal)
        print(f"Image saved: Scanned/myImage{count}.jpg")
        count += 1
