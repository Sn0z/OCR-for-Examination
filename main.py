import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile

# Function to process MCQ image and calculate score
def process_mcq_image(image_path, answer_key):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 11, 4)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubble_contours = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if 20 < w < 50 and 20 < h < 50 and 0.8 < aspect_ratio < 1.2:
            bubble_contours.append((x, y, w, h, c))

    bubble_contours = sorted(bubble_contours, key=lambda x: (x[1], x[0]))
    questions = [bubble_contours[i:i+5] for i in range(0, len(bubble_contours), 5)]

    results = []
    correct = 0

    for q_index, q_bubbles in enumerate(questions):
        q_bubbles = sorted(q_bubbles, key=lambda b: b[0])
        max_intensity = 0
        selected = None
        for i, (_, _, _, _, c) in enumerate(q_bubbles):
            mask = np.zeros(gray.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            total = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
            if total > max_intensity:
                max_intensity = total
                selected = i

        correct_option = answer_key[q_index] if q_index < len(answer_key) else -1
        is_correct = selected == correct_option
        results.append((q_index + 1, selected, correct_option, is_correct))

        color = (0, 255, 0) if is_correct else (0, 0, 255)
        if selected is not None:
            cv2.drawContours(image, [q_bubbles[selected][4]], -1, color, 2)

        if is_correct:
            correct += 1

    score = (correct / len(questions)) * 100 if questions else 0
    return image, results, score

# Streamlit UI
st.title("MCQ Exam Grader")
st.write("Upload a scanned MCQ exam sheet to grade it.")

uploaded_file = st.file_uploader("Upload Exam Image", type=["png", "jpg", "jpeg"])

# Sample answer key (can be replaced with user input)
default_answer_key = [0, 1, 2, 3, 4, 1, 0, 3, 2, 4]  # Example key for 10 questions

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Grading the exam..."):
        annotated_image, results, score = process_mcq_image(tmp_path, default_answer_key)

    st.image(annotated_image, caption="Graded Answer Sheet", channels="BGR")
    st.success(f"Score: {score:.2f}%")

    st.subheader("Results Table")
    for q_num, selected, correct_opt, is_correct in results:
        st.write(f"Question {q_num}: Selected Option {chr(65 + selected) if selected is not None else 'None'} | Correct: {chr(65 + correct_opt)} | {'✅' if is_correct else '❌'}")

    os.remove(tmp_path)
