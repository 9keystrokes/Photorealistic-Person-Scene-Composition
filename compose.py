import cv2
import numpy as np
from rembg import remove
from skimage import exposure
import os
import argparse

# Parse command-line options
parser = argparse.ArgumentParser(description='Compose person into scene')
parser.add_argument('--no-shadow', action='store_true', help='Disable synthetic shadow overlay')
args = parser.parse_args()
apply_shadow = not args.no_shadow

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
inputs_dir = os.path.join(script_dir, 'inputs')
person_input = os.path.join(inputs_dir, 'person.jpg')
bg_input = os.path.join(inputs_dir, 'background.jpg')
output_dir = os.path.join(script_dir, 'output')

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Verify person image
if not os.path.exists(person_input):
    print(f"Error: person.jpg not found in {inputs_dir}. Please add a front-view photo named 'person.jpg'.")
    exit(1)
# Verify background image
if not os.path.exists(bg_input):
    print(f"Error: background.jpg not found in {inputs_dir}. Please add a background image named 'background.jpg'.")
    exit(1)

# Step 1: Remove background from person photo
with open(person_input, 'rb') as inp:
    input_img = inp.read()
output = remove(input_img)
# Save extracted person with same format as input
person_ext = os.path.splitext(person_input)[1]
person_output_path = os.path.join(output_dir, f'extracted_person{person_ext}')
with open(person_output_path, 'wb') as out:
    out.write(output)

# Load images
person = cv2.imread(person_output_path, cv2.IMREAD_UNCHANGED)
bg = cv2.imread(bg_input, cv2.IMREAD_UNCHANGED)

# Step 2: Match brightness to background
# Convert person and background to RGB
person_rgb = cv2.cvtColor(person[..., :3], cv2.COLOR_BGR2RGB)
bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
# Convert to LAB color space
lab_person = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2LAB)
lab_bg = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2LAB)
# Match luminance channel only
lab_person[..., 0] = exposure.match_histograms(lab_person[..., 0], lab_bg[..., 0], channel_axis=None)
# Convert LAB result back to BGR
matched_rgb = cv2.cvtColor(lab_person, cv2.COLOR_LAB2RGB)
matched_bgr = cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2BGR)

# Store original BGR channels
original_bgr = person[..., :3]
# Blend for natural tone
matched_bgr = cv2.addWeighted(original_bgr, 0.4, matched_bgr, 0.6, 0)
# Gamma correction
gamma_correction = 0.9
inv_gamma_correction = 1.0 / gamma_correction
gamma_lut = np.array([((i / 255.0) ** inv_gamma_correction) * 255 for i in range(256)], dtype="uint8")
matched_bgr = cv2.LUT(matched_bgr, gamma_lut)

# Boost saturation
hsv_image = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv_image[..., 1] = np.clip(hsv_image[..., 1] * 1.1, 0, 255)
matched_bgr = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

# CLAHE for local contrast
lab_clahe = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
lab_clahe[..., 0] = clahe.apply(lab_clahe[..., 0])
matched_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# Warmth and vibrance tweaks
matched_bgr = matched_bgr.astype(np.float32)
matched_bgr[..., 2] = np.clip(matched_bgr[..., 2] * 1.05, 0, 255)
matched_bgr = matched_bgr.astype(np.uint8)

lab_ab = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
lab_ab[..., 1] = np.clip(lab_ab[..., 1] + 5, 0, 255)
lab_ab[..., 2] = np.clip(lab_ab[..., 2] + 3, 0, 255)
matched_bgr = cv2.cvtColor(lab_ab.astype(np.uint8), cv2.COLOR_LAB2BGR)
hsv_final = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv_final[..., 1] = np.clip(hsv_final[..., 1] * 1.03, 0, 255)
matched_bgr = cv2.cvtColor(hsv_final.astype(np.uint8), cv2.COLOR_HSV2BGR)
final_gamma = 0.95
inv_final_gamma = 1.0 / final_gamma
final_gamma_lut = np.array([((i / 255.0) ** inv_final_gamma) * 255 for i in range(256)], dtype='uint8')
matched_bgr = cv2.LUT(matched_bgr, final_gamma_lut)

# Step 2 (continued): Detect shadows in background
gray_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
_, inv_mask = cv2.threshold(gray_bg, 50, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
mask_closed = cv2.morphologyEx(inv_mask, cv2.MORPH_CLOSE, kernel)
blurred_mask = cv2.GaussianBlur(mask_closed, (51,51), 0)
shadow_soft = (blurred_mask > 128).astype(np.uint8) * 255
shadow_hard = cv2.bitwise_and(mask_closed, cv2.bitwise_not(shadow_soft))
# Save binary masks with same format as input and new names
shadow_soft_path = os.path.join(output_dir, f'background_shadow_soft{person_ext}')
shadow_hard_path = os.path.join(output_dir, f'background_shadow_hard{person_ext}')
cv2.imwrite(shadow_soft_path, shadow_soft)
cv2.imwrite(shadow_hard_path, shadow_hard)

# Step 3: Compute light direction
alpha_mask = person[..., 3] / 255.0
person_mask = (alpha_mask > 0)
# Compute 2D centroids
ys_p, xs_p = np.nonzero(person_mask)
ys_s, xs_s = np.nonzero(shadow_soft)
centroid_person = np.array([xs_p.mean(), ys_p.mean()])
centroid_shadow = np.array([xs_s.mean(), ys_s.mean()])
vec2d = centroid_shadow - centroid_person
# Assign a downward z component for 3D direction
vec3d = np.array([vec2d[0], vec2d[1], -min(bg.shape[:2]) * 0.5])
light_dir = vec3d / np.linalg.norm(vec3d)

# Step 4: Generate synthetic shadow
alpha = person[..., 3] / 255.0
# Refine alpha mask: erode then blur to remove white halo
alpha_channel = person[..., 3]
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
alpha_eroded = cv2.erode(alpha_channel, kernel_small, iterations=1)
alpha_blur = cv2.GaussianBlur(alpha_eroded, (5,5), 0)
alpha = alpha_blur.astype(np.float32) / 255.0
h, w = alpha.shape
shadow = np.zeros((h, w), dtype=np.uint8)
# simple projection
shadow_offset_x = int(50 * light_dir[0])
shadow_offset_y = int(50 * light_dir[1])
shadow_mask = (alpha > 0).astype(np.uint8) * 255
shadow_shifted = np.roll(np.roll(shadow_mask, shadow_offset_y, axis=0), shadow_offset_x, axis=1)
shadow_blur = cv2.GaussianBlur(shadow_shifted, (21, 21), 10)
shadow_final = (shadow_blur * 0.5).astype(np.uint8)

# Composite shadow and person
bg_composite = bg.copy()
# Place synthetic shadow under person if enabled
x_offset, y_offset = 100, bg.shape[0] - h - 50
shadow_bg = bg.copy()
if apply_shadow:
    x1, y1 = max(0, x_offset+shadow_offset_x), max(0, y_offset+shadow_offset_y)
    x2, y2 = x1+w, y1+h
    shadow_roi = shadow_final[0:(y2-y1), 0:(x2-x1)]
    roi_bg = shadow_bg[y1:y2, x1:x2]
    shadow_color = cv2.cvtColor(shadow_roi, cv2.COLOR_GRAY2BGR)
    shadow_bg[y1:y2, x1:x2] = cv2.addWeighted(roi_bg, 1, shadow_color, 0.5, 0)
else:
    shadow_bg = bg.copy()

# Overlay person with alpha blending with clamping to avoid out-of-bounds
bg_crop = shadow_bg
img_h, img_w = bg_crop.shape[:2]
x1 = max(0, x_offset)
y1 = max(0, y_offset)
x2 = min(img_w, x1 + w)
y2 = min(img_h, y1 + h)
h_crop = y2 - y1
w_crop = x2 - x1
if h_crop > 0 and w_crop > 0:
    for c in range(3):
        bg_crop[y1:y2, x1:x2, c] = (
            matched_bgr[0:h_crop, 0:w_crop, c] * alpha[0:h_crop, 0:w_crop]
            + bg_crop[y1:y2, x1:x2, c] * (1 - alpha[0:h_crop, 0:w_crop])
        )
else:
    print("Warning: Person ROI has zero area, skipping overlay")

# Save final composite image with same format as input
bg_ext = os.path.splitext(bg_input)[1]
final_filename = f"final_composite{bg_ext}"
final_path = os.path.join(output_dir, final_filename)
cv2.imwrite(final_path, bg_crop)
print(f'Final composite saved to {final_path}')
