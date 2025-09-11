import os
import re
import numpy as np
import cv2
import os
import re
import numpy as np
import cv2

directory = "./tmp/expect"  

pattern = re.compile(r"_agent_1_channel_(\d+)\.png")
image_files = sorted(
    [file for file in os.listdir(directory) if pattern.match(file)],
    key=lambda x: int(pattern.match(x).group(1))
)

l = 6
grid_size = (l, l)
num_images = len(image_files)

if num_images == 0:
    raise ValueError("No images found matching the pattern.")

def remove_white_background(img, threshold=240):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)  
    coords = cv2.findNonZero(255 - mask)  
    x, y, w, h = cv2.boundingRect(coords)  
    return img[y:y+h, x:x+w]  

first_img = cv2.imread(os.path.join(directory, image_files[0]), cv2.IMREAD_UNCHANGED)
if first_img.shape[-1] == 4:
    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGRA2BGR)
first_img = remove_white_background(first_img)
img_h, img_w = first_img.shape[:2]

canvas_h = grid_size[0] * img_h
canvas_w = grid_size[1] * img_w

canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

for idx, file in enumerate(image_files[:l*l]):
    if idx >= grid_size[0] * grid_size[1]:  
        break
    row = idx // grid_size[1]
    col = idx % grid_size[1]
    img = cv2.imread(os.path.join(directory, file), cv2.IMREAD_UNCHANGED)
    
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = remove_white_background(img) 

    img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_AREA)

    canvas[row * img_h:(row + 1) * img_h, col * img_w:(col + 1) * img_w] = img

output_path = os.path.join(directory, "combined_grid.png")
cv2.imwrite(output_path, canvas)
print(f"Saved combined image at: {output_path}")
