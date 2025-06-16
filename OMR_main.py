import cv2
import numpy as np
import os
import streamlit as st
import utility  

# Parameters
heightImg = 700
widthImg = 700
save_dir = "Scanned"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

st.title("Automatic MCQ OMR Grader")

uploaded_ans_key = st.file_uploader("Upload Correct Answer Key Sheet (Image)", type=["jpg", "jpeg", "png"])
uploaded_student = st.file_uploader("Upload Student Answer Sheet (Image)", type=["jpg", "jpeg", "png"])


def extract_answer_key(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    imgCanny = cv2.dilate(imgCanny, kernel, iterations=1)

    contours, _ = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectCon = utility.rectContour(contours)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)

    if len(rectCon) < 1:
        raise Exception("No answer area found in correct answer sheet.")

    answerContour = rectCon[0]
    biggestPoints = utility.reorder(utility.getCornerPoints(answerContour))
    pts1 = np.float32(biggestPoints)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    boxes = utility.splitBoxes(imgThresh)

    total_boxes = len(boxes)
    for c in [5, 4, 3]:
        if total_boxes % c == 0:
            choices = c
            questions = total_boxes // c
            break
    else:
        raise Exception("Unable to determine number of questions/choices.")

    myPixelVal = np.zeros((questions, choices))
    for i in range(total_boxes):
        row = i // choices
        col = i % choices
        myPixelVal[row][col] = cv2.countNonZero(boxes[i])

    ans = []
    threshold = 0.5 * np.max(myPixelVal)

    for x in range(questions):
        row = myPixelVal[x]
        marked = np.where(row >= threshold)[0]
        if len(marked) == 1:
            ans.append(int(marked[0]))
        elif len(marked) == 0:
            raise Exception(f"Question {x+1} in the answer key has no answer marked.")
        else:
            raise Exception(f"Question {x+1} in the answer key has multiple answers marked. Only one is allowed.")

    return questions, choices, ans


if uploaded_ans_key:
    st.image(uploaded_ans_key, caption="Correct Answer Sheet", use_container_width=True)
    ans_path = f"temp_key_{uploaded_ans_key.name}"
    with open(ans_path, "wb") as f:
        f.write(uploaded_ans_key.read())

    try:
        questions, choices, ans = extract_answer_key(ans_path)
        st.success(f"Extracted {questions} questions with {choices} choices each.\nAnswer Key: {ans}")
    except Exception as e:
        st.error(f"Error reading answer key: {e}")
    finally:
        os.remove(ans_path)

if uploaded_student and 'ans' in locals():
    st.image(uploaded_student, caption="Student Answer Sheet", use_container_width=True)
    stu_path = f"temp_student_{uploaded_student.name}"
    with open(stu_path, "wb") as f:
        f.write(uploaded_student.read())

    try:
        img = cv2.imread(stu_path)
        img = cv2.resize(img, (widthImg, heightImg))
        imgFinal = img.copy()

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

        biggestPoints = utility.reorder(utility.getCornerPoints(answerContour))
        pts1 = np.float32(biggestPoints)
        pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

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
            elif len(marked) == 0:
                raise Exception(f"Question {x+1} in the student sheet is left blank.")
            else:   
                raise Exception(f"Question {x+1} in the student sheet has multiple answers marked. Only one is allowed.")

        score = (sum(grading) / questions) * 100
        text = f"{int(score)}%"

        imgRawDrawings = np.zeros_like(imgWarpColored)
        utility.showAnswers(imgRawDrawings, myIndex, grading, ans)
        imgInvWarp = cv2.warpPerspective(imgRawDrawings, cv2.getPerspectiveTransform(pts2, pts1), (widthImg, heightImg))

        gradePoints = utility.reorder(utility.getCornerPoints(gradeContour))
        ptsG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
        ptsG1 = np.float32(gradePoints)
        matrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)

        imgRawGrade = np.zeros((150, 325, 3), np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 3, 3)
        cv2.putText(imgRawGrade, text, ((325 - tw) // 2, (150 + th) // 2),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)

        imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, matrixG, (widthImg, heightImg))

        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
        imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

        st.success(f"Grading Complete! Score: {int(score)}%")
        st.image(cv2.cvtColor(imgFinal, cv2.COLOR_BGR2RGB), caption="Final Result", channels="RGB", use_container_width=True)

        # Save the final result
        result_filename = os.path.join(save_dir, f"graded_{uploaded_student.name}.png")
        cv2.imwrite(result_filename, imgFinal)

        # Convert image to bytes and offer for download
        is_success, buffer = cv2.imencode(".png", imgFinal)
        if is_success:
            st.download_button(
                label="Download Graded Sheet",
                data=buffer.tobytes(),
                file_name=f"graded_{uploaded_student.name}.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"Error processing student sheet: {e}")
    finally:
        os.remove(stu_path)
