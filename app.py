from flask import Flask, render_template, request
import csv
import os

app = Flask(__name__)

# ---------------------------
# Dummy Sentiment Predictor
# (Replace with your real model later)
# ---------------------------
def predict_sentiment(title, review):
    text = (title + " " + review).lower()

    negative_words = ["bad", "terrible", "worst", "poor", "hate", "awful"]
    positive_words = ["good", "excellent", "best", "love", "great", "amazing"]

    if any(word in text for word in negative_words):
        return "Negative"
    return "Positive"


# ---------------------------
# Home + Prediction
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    table_data = None

    # SINGLE PREDICTION
    if request.method == "POST" and "csv_file" not in request.files:
        title = request.form.get("title_text")
        review = request.form.get("review_text")

        result = predict_sentiment(title, review)

    # BATCH CSV PREDICTION
    if request.method == "POST" and "csv_file" in request.files:
        file = request.files["csv_file"]

        if file.filename.endswith(".csv"):
            file_path = os.path.join("uploaded.csv")
            file.save(file_path)

            table_data = []
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)  # expects: label,title,review
                for row in reader:
                    predicted = predict_sentiment(row["title"], row["review"])
                    table_data.append({
                        "title": row["title"],
                        "review": row["review"],
                        "Predicted Sentiment": predicted
                    })

    return render_template(
        "index.html",
        result=result,
        table_data=table_data
    )


if __name__ == "__main__":
    app.run(debug=True)
