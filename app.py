from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config["ALLOWED_EXTENTIONS"] = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return "." in filename and \
        filename.split(".", 1)[1] in app.config["ALLOWED_EXTENTIONS"]

model = load_model("detect.h5", compile=False)
with open("labels.txt", "r") as file:
    labels = file.read().splitlines() 


@app.route("/")
def index():
     return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200



@app.route("/prediction", methods={"GET", "POST"})
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save("static/uploads", filename)
            return "Saved"
        else:
            return jsonify({
                "status":{
                    "code":400,
                    "message":"Client side error"
                },
                 "data": None
            }), 400                  
        
    else:
            return jsonify({
                "status": {
                    "code": 405,
                   "message": "Method not allowed"      
                },
                "data": None
            }), 405 

if __name__ == "__main__":
    app.run()