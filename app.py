from flask import Flask, request, render_template
from src.model import TextClassifier
from src.pipeline import TextClassificationPipeline

from config import MODEL_PATH, DATA_PATH

app = Flask(__name__)

model_path: str = MODEL_PATH
classifier = TextClassifier(model_path=model_path)

try:
    classifier.load()
except FileNotFoundError:
    print("[!] Модель не найдена. Запускаем pipeline...")
    pipeline = TextClassificationPipeline(data_path=DATA_PATH)
    pipeline.run(evaluate=False)
    classifier.load()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        text = request.form["text"]
        prediction = classifier.predict([text])[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
