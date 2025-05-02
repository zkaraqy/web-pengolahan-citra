from flask import Flask, render_template

app = Flask(__name__)

app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/thresholding")
def thresholding():
    return render_template("thresholding.html")

@app.route("/edge_detection")
def edge_detection():
    return render_template("edge_detection.html")

if __name__ == "__main__":
    app.run(debug=True)