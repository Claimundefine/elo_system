
import easyocr
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# 2. Initialize a reader for the languages you need (e.g. English)
reader = easyocr.Reader(['en'])  # this will download model files on first run

# 3. Read text from an image file
#    You can also pass a NumPy array (e.g. from cv2.imread)
results = reader.readtext('example.png')

img = cv2.imread('example.png')

all_boxes = {
    "col1": [ ((739, 84), (813, 160)), ((840, 85), (1025, 161)),  ((1053, 90), (1118, 158))]
}


horizontal_list = []
for col_name, rects in all_boxes.items():
    for (x1, y1), (x2, y2) in rects:
        horizontal_list.append([x1, x2, y1, y2])

print(horizontal_list)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

results = reader.recognize(
    gray,                      # positional image arg
    horizontal_list=horizontal_list,
    free_list=[],             # explicitly empty
    detail=1,
    paragraph=False
)

print(results)


for bbox, text, confidence in results:
    print(f"Detected “{text}” (conf {confidence:.2f}) in box {bbox}")
