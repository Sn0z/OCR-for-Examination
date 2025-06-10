import cv2
import numpy as np
import os
import streamlit as st
import utility  # your custom utility functions

# Parameters
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]
save_dir = "Scanned"

# Ensure save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Streamlit UI
st.title("MCQ Sheet Grader")
uploaded_file = st.file_uploader("Upload an MCQ answer sheet image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    temp_path = f"temp_{uploaded_file.name}"

    # Save file temporarily
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Process the image
    img = cv2.imread(temp_path)
    img = cv2.resize(img, (widthImg, heightImg))
    imgFinal = img.copy()
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)

    try:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, 30, 100)
        kernel = np.ones((5, 5), np.uint8)
        imgCanny = cv2.dilate(imgCanny, kernel, iterations=1)

        contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rectCon = utility.rectContour(contours)
        rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)

        answerContour = rectCon[0]
        gradeContour = rectCon[1]

        biggestPoints = utility.getCornerPoints(answerContour)
        biggestPoints = utility.reorder(biggestPoints)
        pts1 = np.float32(biggestPoints)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        gradePoints = utility.getCornerPoints(gradeContour)
        gradePoints = utility.reorder(gradePoints)
        ptsG1 = np.float32(gradePoints)
        ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
        imgGradeDisplay = cv2.warpPerspective(img, matrixG, (325, 150))

        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

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
            marked = np.where(row >= threshold)[0]
            if len(marked) == 1:
                myIndex.append(int(marked[0]))
                grading.append(1 if ans[x] == marked[0] else 0)
            else:
                myIndex.append(-1)
                grading.append(0)

        score = (sum(grading) / questions) * 100
        text = f"{int(score)}%"

        # Render grade
        imgRawDrawings = np.zeros_like(imgWarpColored)
        utility.showAnswers(imgRawDrawings, myIndex, grading, ans)
        imgInvWarp = cv2.warpPerspective(imgRawDrawings, cv2.getPerspectiveTransform(pts2, pts1), (widthImg, heightImg))

        imgRawGrade = np.zeros((150, 325, 3), np.uint8)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 3, 3)
        cv2.putText(imgRawGrade, text, ((325 - text_width) // 2, (150 + text_height) // 2),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, cv2.getPerspectiveTransform(ptsG2, ptsG1), (widthImg, heightImg))

        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

        st.success(f"Grading Complete! Score: {int(score)}%")
        st.image(cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB), caption="Final Result", channels="RGB")

    except Exception as e:
        st.error(f"Failed to process image: {e}")

    # Clean up
    os.remove(temp_path)
