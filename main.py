import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import cv2
import shutil
from ocr.ocr_utils import preprocess_image, extract_text, create_text_mask
import keras_ocr

def main():
    img_dir = 'Before'  
    output_dir = 'After'  

    # Check if the output directory exists and remove it if it does
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create a new output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the OCR pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # List all image files in the input directory, excluding hidden files and .ipynb_checkpoints
    images_list = [im for im in os.listdir(img_dir) if not im.startswith('.') and not im.endswith('.ipynb_checkpoints')]

    for img in images_list:
        print(f"Processing file: {img}")  # Print the name of the file being processed
        image_path = os.path.join(img_dir, img)
        rgb_contrast_img, color_image = preprocess_image(image_path)
        predictions = extract_text(rgb_contrast_img, pipeline)
        white_image = create_text_mask(color_image, predictions)
        output_filename = os.path.splitext(img)[0] + '_text_only.png'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, white_image)

    print('Text has been extracted and placed on white background images in the output directory.')

if __name__ == "__main__":
    main()
