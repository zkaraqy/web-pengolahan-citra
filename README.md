# Advanced Digital Image Processing Web Application

This project is a web-based application for advanced digital image processing, developed as the Final Project (UAS) for the Digital Image Processing course at university. The application enables users to perform a variety of morphological and edge detection operations on images through an intuitive web interface.

## What is this project?
This website allows users to upload images and apply a wide range of image processing techniques, including:
- **Basic Morphological Operations**: Erosion, Dilation
- **Advanced Morphological Operations**: Skeletonization, Thickening, Filling, Convex Hull, Boundary Extraction, Thinning, Pruning (Spur Removal)
- **Thresholding**: Convert images to binary using adjustable thresholds
- **Edge Detection**: Sobel, Prewitt, Roberts, and Canny operators

## Why was this project created?
This project was created as a practical implementation of the concepts learned in the Digital Image Processing course. It serves as a demonstration of understanding and applying various image processing algorithms, and provides a user-friendly platform for experimenting with these techniques. The project also aims to help other students and practitioners learn about digital image processing through hands-on interaction.

## Who developed this project?
- **Author:** Muhammad Azka Raki
- **Course:** Digital Image Processing (Pengolahan Citra)
- **Purpose:** Final Project (UAS)

## How does the website work?
The website is built using Python, Flask, OpenCV, and Bootstrap. Users can:
1. Access the website via a browser
2. Choose the desired image processing method (Morphology, Thresholding, Edge Detection, etc.)
3. Upload an image
4. Adjust parameters as needed (e.g., threshold value, kernel size, operation type)
5. Process the image and view the results instantly
6. Download the processed image for further use

All image processing is performed on the server using OpenCV, and results are displayed in real-time on the web interface.

## Installation & Local Running Guide

### Prerequisites
- Python 3.10 or higher
- Git (optional, for cloning the repository)

### 1. Clone the Repository
```bash
git clone https://github.com/zkaraqy/web-pengolahan-citra.git
cd web-pengolahan-citra
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r dev-requirements.txt
```

### 5. Run the Flask Application
```bash
python app.py
```

### 6. Open the Website
Open your browser and go to:
```
http://127.0.0.1:5000/
```

## Project Structure
```
web-pengolahan-citra/
├── app.py                 # Main Flask application
├── static/
│   ├── css/               # CSS files (Bootstrap, etc.)
│   ├── js/                # JavaScript files
│   ├── img/               # Static images (logo, etc.)
│   ├── uploads/           # Uploaded images
│   └── processed/         # Processed images
├── templates/
│   ├── base.html          # Base template
│   ├── home.html          # Home page
│   ├── morfologi.html     # Basic morphology page
│   ├── morfologi_lanjutan.html # Advanced morphology page
│   ├── thresholding.html  # Thresholding page
│   └── edge_detection.html # Edge detection page
├── requirements.txt       # Main dependencies
├── dev-requirements.txt   # Development dependencies
├── packages.txt           # System dependencies (for cloud deployment)
└── README.md              # Project documentation
```

## License
This project is for educational purposes only.

---