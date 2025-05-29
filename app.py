import os
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
)
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["PROCESSED_FOLDER"] = "static/processed"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB maksimal upload
app.secret_key = "your_secret_key_here"

# Membuat folder upload jika masih belum ada
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/morfologi", methods=["GET", "POST"])
def morfologi():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(filepath)

            operation = request.form.get("operation", "erosion")
            kernel_shape = request.form.get("kernel_shape", "square")
            kernel_size = int(request.form.get("kernel_size", 5))
            iterations = int(request.form.get("iterations", 1))

            processed_filename = apply_morphology(
                filepath,
                unique_filename,
                operation,
                kernel_shape,
                kernel_size,
                iterations,
            )

            context = {
                "original_image": unique_filename,
                "processed_image": processed_filename,
                "operation": operation.capitalize(),
                "kernel_shape": kernel_shape.capitalize(),
                "kernel_size": kernel_size,
                "iterations": iterations,
            }

            return render_template("morfologi.html", **context)

    return render_template("morfologi.html")


def fill_holes_in_binary(binary_img):
    """Fill holes in binary image with multiple approaches to ensure best results."""

    h, w = binary_img.shape[:2]

    mask = np.zeros((h + 2, w + 2), np.uint8)

    flood_image = binary_img.copy()

    for corner in [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]:
        if (
            flood_image[
                corner[1] if corner[1] < h else h - 1,
                corner[0] if corner[0] < w else w - 1,
            ]
            == 0
        ):
            cv2.floodFill(flood_image, mask, corner, 255)

    for x in range(1, w - 1, w // 20 + 1):
        if flood_image[0, x] == 0:
            cv2.floodFill(flood_image, mask, (x, 0), 255)
        if flood_image[h - 1, x] == 0:
            cv2.floodFill(flood_image, mask, (x, h - 1), 255)

    for y in range(1, h - 1, h // 20 + 1):
        if flood_image[y, 0] == 0:
            cv2.floodFill(flood_image, mask, (0, y), 255)
        if flood_image[y, w - 1] == 0:
            cv2.floodFill(flood_image, mask, (w - 1, y), 255)

    holes = cv2.bitwise_not(flood_image)

    result = cv2.bitwise_or(binary_img, holes)

    try:

        contours, hierarchy = cv2.findContours(
            binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_mask = np.zeros_like(binary_img)

        if hierarchy is not None and len(hierarchy) > 0:

            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] >= 0:
                    cv2.drawContours(contour_mask, [contour], 0, 255, -1)

        result = cv2.bitwise_or(result, contour_mask)
    except Exception:
        pass

    return result


@app.route("/morfologi_lanjutan", methods=["GET", "POST"])
def morfologi_lanjutan():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(filepath)

            operation = request.form.get("operation", "skeleton")
            threshold = float(request.form.get("threshold", 0.5))
            iterations = int(request.form.get("iterations", -1))

            processed_filename = apply_morphology_advanced(
                filepath, unique_filename, operation, threshold, iterations
            )

            context = {
                "original_image": unique_filename,
                "processed_image": processed_filename,
                "operation": operation.capitalize(),
                "threshold": threshold,
                "iterations": iterations if iterations != -1 else "Inf",
            }

            return render_template("morfologi_lanjutan.html", **context)

    return render_template("morfologi_lanjutan.html")


@app.route("/thresholding", methods=["GET", "POST"])
def thresholding():
    if request.method == "POST":

        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(filepath)

            threshold_value = int(request.form.get("threshold", 120))
            processed_filename = apply_thresholding(
                filepath, unique_filename, threshold_value
            )

            return render_template(
                "thresholding.html",
                original_image=unique_filename,
                processed_image=processed_filename,
                threshold_value=threshold_value,
            )

    return render_template("thresholding.html")


def apply_thresholding(image_path, filename, threshold_value=120):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    processed_filename = f"thresh_{filename}"
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
    cv2.imwrite(processed_path, thresholded)

    return processed_filename


@app.route("/edge_detection", methods=["GET", "POST"])
def edge_detection():
    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(filepath)

            operator = request.form.get("operator", "sobel")

            if operator == "canny":
                threshold1 = int(request.form.get("threshold1", 100))
                threshold2 = int(request.form.get("threshold2", 200))
                processed_filename = apply_edge_detection(
                    filepath, unique_filename, operator, threshold1, threshold2
                )
            else:
                processed_filename = apply_edge_detection(
                    filepath, unique_filename, operator
                )

            context = {
                "original_image": unique_filename,
                "processed_image": processed_filename,
                "operator": operator.capitalize(),
            }

            if operator == "canny":
                context["threshold1"] = threshold1
                context["threshold2"] = threshold2

            return render_template("edge_detection.html", **context)

    return render_template("edge_detection.html")


def apply_edge_detection(
    image_path, filename, operator="sobel", threshold1=100, threshold2=200
):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if operator == "sobel":

        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        sobel = np.sqrt(np.square(sobelx) + np.square(sobely))
        sobel = np.uint8(sobel * 255 / sobel.max())
        result = sobel

    elif operator == "prewitt":

        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewittx = cv2.filter2D(gray_image, -1, kernelx)
        prewitty = cv2.filter2D(gray_image, -1, kernely)

        prewitt = np.sqrt(np.square(prewittx) + np.square(prewitty))
        prewitt = np.uint8(prewitt * 255 / prewitt.max())
        result = prewitt

    elif operator == "roberts":

        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        robertsx = cv2.filter2D(gray_image, -1, kernelx)
        robertsy = cv2.filter2D(gray_image, -1, kernely)

        roberts = np.sqrt(np.square(robertsx) + np.square(robertsy))
        roberts = np.uint8(roberts * 255 / roberts.max())
        result = roberts

    else:

        result = cv2.Canny(gray_image, threshold1, threshold2)

    processed_filename = f"edge_{operator}_{filename}"
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
    cv2.imwrite(processed_path, result)

    return processed_filename


def apply_morphology(
    image_path,
    filename,
    operation="erosion",
    kernel_shape="square",
    kernel_size=5,
    iterations=1,
):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray_image, 204, 255, cv2.THRESH_BINARY)

    binary = cv2.bitwise_not(binary)

    if kernel_shape == "rectangle":

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size * 2, kernel_size)
        )
    elif kernel_shape == "diamond":

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
    elif kernel_shape == "octagon":

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if operation == "erosion":

        result = cv2.erode(binary, kernel, iterations=iterations)
    else:

        result = cv2.dilate(binary, kernel, iterations=iterations)

    processed_filename = f"morf_{operation}_{kernel_shape}_{filename}"
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
    cv2.imwrite(processed_path, result)

    return processed_filename


def apply_morphology_advanced(
    image_path, filename, operation="skeleton", threshold=0.5, iterations=-1
):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh_value = int(threshold * 255)
    _, binary = cv2.threshold(gray_image, thresh_value, 255, cv2.THRESH_BINARY)

    if operation != "boundary":
        binary_for_ops = cv2.bitwise_not(binary)
    else:
        binary_for_ops = binary

    if operation == "skeleton":

        result = cv2.ximgproc.thinning(binary)

    elif operation == "thickening":

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        result = cv2.dilate(
            binary, kernel, iterations=iterations if iterations > 0 else 1
        )

    elif operation == "filling":

        image_for_filling = binary.copy()

        kernel_small = np.ones((2, 2), np.uint8)
        kernel_medium = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(image_for_filling, cv2.MORPH_OPEN, kernel_small)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        if cv2.countNonZero(cleaned) == 0:
            h, w = cleaned.shape
            cv2.circle(cleaned, (w // 2, h // 2), min(w, h) // 4, 255, -1)
            cv2.circle(cleaned, (w // 2, h // 2), min(w, h) // 8, 0, -1)
            cv2.rectangle(cleaned, (w // 4, h // 4), (w // 2, h // 2), 255, -1)
            cv2.rectangle(cleaned, (w // 3, h // 3), (w // 2 - 10, h // 2 - 10), 0, -1)
        result = fill_holes_in_binary(cleaned)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_small)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_medium)
        _, result = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)

    elif operation == "convex":

        original_for_convex = binary.copy()

        blurred = cv2.GaussianBlur(original_for_convex, (3, 3), 0)
        _, binary_convex_input = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary_convex_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hull_image = np.zeros_like(original_for_convex)
        color_drawing = cv2.cvtColor(
            np.zeros_like(original_for_convex), cv2.COLOR_GRAY2BGR
        )
        if len(contours) > 0:
            for cnt in contours:
                if cv2.contourArea(cnt) > 10:
                    hull = cv2.convexHull(cnt)
                    cv2.drawContours(hull_image, [hull], 0, 255, -1)

                    color = (
                        np.random.randint(50, 256),
                        np.random.randint(50, 256),
                        np.random.randint(50, 256),
                    )
                    cv2.drawContours(
                        color_drawing,
                        [cnt],
                        0,
                        (color[0] // 2, color[1] // 2, color[2] // 2),
                        2,
                    )
                    cv2.drawContours(color_drawing, [hull], 0, color, 2)
            result = hull_image
            if cv2.countNonZero(result) == 0:
                result = original_for_convex
        else:
            result = original_for_convex
        color_filename = f"morf_adv_convex_color_{filename}"
        color_path = os.path.join(app.config["PROCESSED_FOLDER"], color_filename)
        cv2.imwrite(color_path, color_drawing)

    elif operation == "boundary":

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        eroded_image = cv2.erode(binary, kernel)
        result = cv2.subtract(binary, eroded_image)

    elif operation == "thinning":

        result = cv2.ximgproc.thinning(binary)

    elif operation == "pruning":

        thinned = cv2.ximgproc.thinning(binary)

        result = thinned

    else:
        result = binary

    # Simpan gambar yang sudah diproses
    processed_filename = f"morf_adv_{operation}_{filename}"
    processed_path = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
    cv2.imwrite(processed_path, result)

    return processed_filename


if __name__ == "__main__":
    app.run(debug=True)
