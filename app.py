from flask import Flask, request, render_template
from src.model import TextClassifier
import pandas as pd


app = Flask(__name__)

classifier = TextClassifier()
data = pd.read_csv("data/data.csv")
X = data["text"]
y = data["label"]
classifier.train(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]
        prediction = classifier.predict([text])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
