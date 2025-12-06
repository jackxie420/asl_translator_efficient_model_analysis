import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import string

data_path = Path("data/asl_alphabet_train")
output_dir = "extracted_features"

print("Initializing MediaPipe...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs(output_dir, exist_ok=True)

if not data_path.exists():
    print(f"Error: Dataset folder not found at {data_path}")
    print(f"Looking for: {data_path.absolute()}")
    exit(1)

letter_folders = [f for f in data_path.iterdir() if f.is_dir() and f.name in string.ascii_uppercase]
letter_folders.sort()

print(f"\nFound {len(letter_folders)} letter folders")
print(f"Processing train folder: {data_path}")

for letter_folder in tqdm(letter_folders, desc="Processing letters"):
    letter_name = letter_folder.name
    print(f"\nProcessing letter: {letter_name}")
    
    image_files = list(letter_folder.glob("*.jpg")) + list(letter_folder.glob("*.JPG")) + \
                  list(letter_folder.glob("*.jpeg")) + list(letter_folder.glob("*.JPEG"))
    
    if not image_files:
        print(f"  No image files found in {letter_name}")
        continue
    
    print(f"  Found {len(image_files)} images")
    
    all_data = []
    successful_extractions = 0
    
    for image_path in tqdm(image_files, desc=f"  {letter_name}", leave=False):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            row_data = {}
            
            for idx, landmark in enumerate(landmarks.landmark):
                row_data[f'landmark_{idx}_x'] = landmark.x
                row_data[f'landmark_{idx}_y'] = landmark.y
                row_data[f'landmark_{idx}_z'] = landmark.z
            
            all_data.append(row_data)
            successful_extractions += 1
    
    if all_data:
        df = pd.DataFrame(all_data)
        output_file = os.path.join(output_dir, f"{letter_name.lower()}.csv")
        df.to_csv(output_file, index=False, header=False)
        print(f"  Saved {successful_extractions} coordinates to {output_file}")
        print(f"  Shape: {df.shape}")
    else:
        print(f"  No hand landmarks detected in any images for {letter_name}")

print(f"\n\nExtraction complete! CSV files saved to {output_dir}/")
print("Each letter has its own CSV file containing all extracted coordinates.")

