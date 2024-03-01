import cv2
import numpy as np
import keras_ocr

def preprocess_image(image_path):
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    denoised_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(denoised_image)
    rgb_contrast_img = cv2.cvtColor(contrast_img, cv2.COLOR_GRAY2RGB)
    return rgb_contrast_img, color_image

def extract_text(rgb_contrast_img, pipeline):
    predictions = pipeline.recognize([rgb_contrast_img])[0]
    return predictions

def create_text_mask(color_image, predictions):
    white_image = np.full_like(color_image, 255)
    for text, box in predictions:
        x_min, y_min = np.min(box, axis=0).astype(int)
        x_max, y_max = np.max(box, axis=0).astype(int)
        text_region = color_image[y_min:y_max, x_min:x_max]
        gray_text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
        _, text_mask = cv2.threshold(gray_text_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_mask = cv2.bitwise_not(text_mask)
        white_image[y_min:y_max, x_min:x_max][text_mask == 255] = text_region[text_mask == 255]
    return white_image
