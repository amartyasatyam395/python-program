import cv2
import numpy as np
from tkinter import Tk, filedialog
import os

# --------- FIXED OUTPUT FOLDER ----------
OUTPUT_FOLDER = r"D:\vs code"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------- FILE PICKER ----------
Tk().withdraw()
image_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
)

if not image_path:
    print("No image selected!")
    exit()

# --------- LOAD IMAGE ----------
img = cv2.imread(image_path)
if img is None:
    print("Failed to load image!")
    exit()

# --------- STEP 1: DENOISE ----------
denoised = cv2.fastNlMeansDenoisingColored(
    img, None, 10, 10, 7, 21
)

# --------- STEP 2: CONTRAST ----------
lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
l = clahe.apply(l)

enhanced_lab = cv2.merge((l, a, b))
contrast_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

# --------- STEP 3: SHARPEN ----------
kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
sharpened = cv2.filter2D(contrast_img, -1, kernel)

# --------- SAVE OUTPUT ----------
filename = os.path.basename(image_path)
name, ext = os.path.splitext(filename)
output_path = os.path.join(OUTPUT_FOLDER, f"{name}_enhanced{ext}")

cv2.imwrite(output_path, sharpened)

# --------- DISPLAY ----------
cv2.imshow("Original Image", img)
cv2.imshow("Enhanced Image", sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Enhanced image saved to:")
print(output_path)
