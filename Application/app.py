# flask application for loading ml model and predicting the value

import os
import pickle
import numpy as np
from flask import (
    Flask,
    render_template,
    url_for,
    request,
    session,
    redirect,
    flash,
    url_for,
)
from pymongo import MongoClient
from dotenv import load_dotenv

app = Flask(__name__)
load_dotenv()

app.secret_key = "finance"
mongoURI = os.getenv(
    "YOUR_MONGO_URL")
MONGODB_URI = mongoURI
client = MongoClient(MONGODB_URI)
db = client["users"]

# load the model from disk
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('login.html')


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"]
        pid = request.form["pid"]
        password = request.form["password"]
        db.users.insert_one({"email": email, "pid": pid, "password": password})
        return redirect(url_for("home"))
    return render_template("signup.html")


@app.route("/home")
def home():
    if "pid" in session:
        return render_template("index.html")
    else:
        return redirect("/login")


@app.route("/analysis")
def analysis():
    if "pid" in session:
        return render_template("analysis.html")
    else:
        return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        pid = request.form["pid"]
        password = request.form["password"]
        user = db.users.find_one({"pid": pid, "password": password})
        if user is not None:
            session["pid"] = pid
            return redirect(url_for("home"))
        else:
            flash("Invalid Patient ID or Password. Please Try again.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("pid", None)
    return redirect(url_for("login"))


@app.route('/predict', methods=['POST'])
def predict():
    # get the data from the POST request
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # predict the value
    prediction = model.predict(final_features)
    # output = round(prediction[0])
    if (prediction == [0]):
        prediction = "You will not become Financially Independent"
    else:
        prediction = "You will become Financially Independent"

    return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
