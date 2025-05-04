import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB maksimal upload
app.secret_key = 'your_secret_key_here'

# Membuat folder upload jika masih belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/thresholding", methods=['GET', 'POST'])
def thresholding():
    if request.method == 'POST':
        # Cek jika request post punya file
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['image']
        
        # Jika pengguna tidak memilih file, browser juga akan mengirim bagian kosong tanpa nama file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Membuat nama file unik untuk mencegah penimpaan
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Melakukan thresholding pada gambar
            threshold_value = int(request.form.get('threshold', 120))
            processed_filename = apply_thresholding(filepath, unique_filename, threshold_value)
            
            # Berikan nama file asli dan yang sudah diproses ke template
            return render_template('thresholding.html', original_image=unique_filename, processed_image=processed_filename, threshold_value=threshold_value)
    
    return render_template("thresholding.html")

def apply_thresholding(image_path, filename, threshold_value=120):
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Terapkan thresholding biner
    _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Simpan gambar yang sudah diproses
    processed_filename = f"thresh_{filename}"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, thresholded)
    
    return processed_filename

@app.route("/edge_detection", methods=['GET', 'POST'])
def edge_detection():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['image']
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Ambil parameter edge detection dari form
            operator = request.form.get('operator', 'sobel')
            
            # Parameter tambahan untuk Canny
            if operator == 'canny':
                threshold1 = int(request.form.get('threshold1', 100))
                threshold2 = int(request.form.get('threshold2', 200))
                processed_filename = apply_edge_detection(filepath, unique_filename, operator, threshold1, threshold2)
            else:
                processed_filename = apply_edge_detection(filepath, unique_filename, operator)
            
            # Berikan nama file dan parameter ke template
            context = {
                'original_image': unique_filename,
                'processed_image': processed_filename,
                'operator': operator.capitalize()
            }
            
            if operator == 'canny':
                context['threshold1'] = threshold1
                context['threshold2'] = threshold2
                
            return render_template('edge_detection.html', **context)
    
    return render_template("edge_detection.html")

def apply_edge_detection(image_path, filename, operator='sobel', threshold1=100, threshold2=200):
    image = cv2.imread(image_path)
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Terapkan operator deteksi tepi berdasarkan operator yang dipilih
    if operator == 'sobel':
        # Terapkan operator sobel
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        # Hitung magnitudo gradien
        sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        sobel = np.uint8(sobel * 255 / sobel.max())
        result = sobel
        
    elif operator == 'prewitt':
        # Terapkan operator prewitt
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(gray_image, -1, kernelx)
        prewitty = cv2.filter2D(gray_image, -1, kernely)
        # Hitung magnitudo gradien
        prewitt = np.sqrt(np.square(prewittx) + np.square(prewitty))
        prewitt = np.uint8(prewitt * 255 / prewitt.max())
        result = prewitt
        
    elif operator == 'roberts':
        # Terapkan operator roberts
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(gray_image, -1, kernelx)
        robertsy = cv2.filter2D(gray_image, -1, kernely)
        # Hitung magnitudo gradien
        roberts = np.sqrt(np.square(robertsx) + np.square(robertsy))
        roberts = np.uint8(roberts * 255 / roberts.max())
        result = roberts
        
    else:  # 'canny'
        # Terapkan deteksi tepi Canny
        result = cv2.Canny(gray_image, threshold1, threshold2)
    
    # Simpan gambar yang sudah diproses
    processed_filename = f"edge_{operator}_{filename}"
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, result)
    
    return processed_filename

if __name__ == "__main__":
    app.run(debug=True)