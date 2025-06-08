import cv2
import numpy as np
import os
import utility2

# Setup
pathImage = "6.jpg"
heightImg = 700
widthImg = 700
questions = 5
choices = 5
ans = [1, 2, 0, 2, 4]
count = 0
save_dir = "Scanned"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

img = cv2.imread(pathImage)
img = cv2.resize(img, (widthImg, heightImg))
clone = img.copy()

clicked_points = []
region_names = ["Answer Sheet", "Grade Box"]
region_index = 0

def mousePoints(event, x, y, flags, params):
    global clicked_points, region_index
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"Point {len(clicked_points)} for {region_names[region_index]}: ({x}, {y})")
        if len(clicked_points) == 4:
            region_index += 1

# Collect both answer and grade points
all_pts = []
cv2.namedWindow("Select Corners")
cv2.setMouseCallback("Select Corners", mousePoints)

while region_index < 2:
    temp = clone.copy()
    for pt in clicked_points:
        cv2.circle(temp, tuple(pt), 10, (0, 255, 0), cv2.FILLED)

    cv2.putText(temp, f"Select 4 corners for: {region_names[region_index]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Select Corners", temp)
    cv2.waitKey(1)

    if len(clicked_points) == 4:
        all_pts.append(np.array(clicked_points, dtype=np.float32))
        clicked_points = []

cv2.destroyWindow("Select Corners")

# --- Fixed transformation points ---
pts1 = np.array(utility2.reorder(all_pts[0]), dtype=np.float32).reshape(4, 2)
pts2 = np.array([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]], dtype=np.float32)
matrix = cv2.getPerspectiveTransform(pts1, pts2)
imgWarpColored = cv2.warpPerspective(clone, matrix, (widthImg, heightImg))

ptsG1 = np.array(utility2.reorder(all_pts[1]), dtype=np.float32).reshape(4, 2)
ptsG2 = np.array([[0, 0], [325, 0], [0, 150], [325, 150]], dtype=np.float32)
matrixG = cv2.getPerspectiveTransform(ptsG1, ptsG2)
imgGradeDisplay = cv2.warpPerspective(clone, matrixG, (325, 150))

# Threshold the answer area
imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

boxes = utility2.splitBoxes(imgThresh)
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

utility2.showAnswers(imgWarpColored, myIndex, grading, ans)
utility2.drawGrid(imgWarpColored)

imgRawDrawings = np.zeros_like(imgWarpColored)
utility2.showAnswers(imgRawDrawings, myIndex, grading, ans)

invMatrix = cv2.getPerspectiveTransform(pts2, pts1)
imgInvWarp = cv2.warpPerspective(imgRawDrawings, invMatrix, (widthImg, heightImg))

imgRawGrade = np.zeros((150, 325, 3), np.uint8)
cv2.putText(
    imgRawGrade,
    f"{int(score)}%",
    (80, 100),
    cv2.FONT_HERSHEY_COMPLEX,
    3,
    (0, 0, 255),
    3
)

invMatrixG = cv2.getPerspectiveTransform(ptsG2, ptsG1)
imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, heightImg))

imgFinal = cv2.addWeighted(clone, 1, imgInvWarp, 1, 0)
imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

cv2.imshow("Final Result", imgFinal)

# Save on keypress
if cv2.waitKey(0) & 0xFF == ord('s'):
    save_path = os.path.join(save_dir, f"myImage{count}.jpg")
    cv2.imwrite(save_path, imgFinal)
    print(f"Image saved: {save_path}")
    count += 1

cv2.destroyAllWindows()
