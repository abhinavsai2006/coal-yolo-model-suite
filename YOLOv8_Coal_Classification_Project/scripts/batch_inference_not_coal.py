import os
from pathlib import Path
from predict_coal_type import predict_coal_type

MODEL_PATH = 'coal_classification_runs/coal_yolov8_small/weights/best.pt'
IMAGES_DIR = 'data/val/not_coal'
OUTPUT_FILE = 'inference_results_not_coal.csv'

image_extensions = ['.jpg', '.jpeg', '.png']

results = []

for img_file in os.listdir(IMAGES_DIR):
    if any(img_file.lower().endswith(ext) for ext in image_extensions):
        img_path = os.path.join(IMAGES_DIR, img_file)
        try:
            pred, conf = predict_coal_type(MODEL_PATH, img_path)
            results.append(f'{img_file},{pred},{conf:.4f}')
        except Exception as e:
            results.append(f'{img_file},ERROR,{str(e)}')

with open(OUTPUT_FILE, 'w') as f:
    f.write('image,predicted_class,confidence\n')
    for line in results:
        f.write(line + '\n')

print(f'Results saved to {OUTPUT_FILE}')
