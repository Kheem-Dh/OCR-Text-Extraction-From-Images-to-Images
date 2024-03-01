# OCR Text Extraction Project

This project is designed to extract text from images using Optical Character Recognition (OCR) and place the extracted text onto white background images. It is implemented in Python and utilizes libraries such as OpenCV, NumPy, and keras-ocr.

## Features

- Read images from a specified directory.
- Apply image preprocessing techniques like grayscale conversion, Gaussian blur, and CLAHE for contrast adjustment.
- Use keras-ocr to perform OCR on images.
- Extract recognized text and place it on a blank white image.
- Save the output images in a specified directory.

## Prerequisites

Ensure you have Python 3.9.13 or higher installed. The following libraries are also required:
- OpenCV
- keras-ocr

You can install these dependencies via pip:

```bash
pip install -r requirements.txt
```

## Usage
To run the script, navigate to the project's root directory and use the following command:

```bash
python main.py  
```