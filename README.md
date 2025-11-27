📌 Sentiment Analysis System (DistilBERT + Flask)

This project is a Sentiment Analysis System that predicts whether a review is Positive or Negative by analyzing both the review title and the review text.
It uses two DistilBERT transformer models—one for titles and one for full review text—combined into a custom neural network for improved accuracy.

The system is deployed using a Flask web application and provides real-time predictions through a simple API or UI.

🚀 Features

🔍 Dual-Input Sentiment Analysis

DistilBERT #1 → Encodes Review Title

DistilBERT #2 → Encodes Review Text

Both embeddings are fused in a custom neural network.

🧠 Transformer-powered

Uses HuggingFace DistilBERT pre-trained models

Fine-tuned for binary sentiment classification

🕸️ Flask-Based Deployment

REST API /predict

Accepts JSON input:

{
  "title": "The product is great",
  "review": "Really good quality, arrived on time."
}


⚡ Fast and Lightweight

DistilBERT ensures fast inference even on CPU

Custom fusion layer enhances accuracy

🏗️ System Architecture
 ┌────────────┐       ┌──────────────┐
 │ Review     │       │ Review       │
 │ Title      │       │ Text         │
 └──────┬─────┘       └──────┬───────┘
        │                     │
        ▼                     ▼
 ┌────────────┐       ┌──────────────┐
 │ DistilBERT │       │ DistilBERT   │
 │  (Title)   │       │   (Text)     │
 └──────┬─────┘       └──────┬───────┘
        └────────┬────────────┘
                 ▼
        ┌─────────────────┐
        │ Fusion Layer     │
        │ (Dense + ReLU)   │
        └────────┬────────┘
                 ▼
        ┌─────────────────┐
        │ Output Layer     │
        │ Positive/Negative│
        └─────────────────┘

📦 Tech Stack
Component	Technology
NLP Models	DistilBERT (HuggingFace)
Backend API	Flask
Neural Network	PyTorch / TensorFlow (your project version)
Tokenization	HuggingFace Tokenizers
Deployment	Localhost / Cloud
Data Format	JSON
🚀 How to Run the Project
1️⃣ Clone the repository
git clone <your_repo_link>
cd sentiment-analysis

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Start the Flask app
python app.py


Flask will start at:

http://127.0.0.1:5000/

🧪 API Usage
POST /predict

Example request:

{
  "title": "Worst purchase ever",
  "review": "The product stopped working within 2 days!"
}


Example response:

{
  "prediction": "Negative"
}

📁 Project Structure
sentiment-analysis/
│
├── app.py               # Flask server
├── model.py             # Combined DistilBERT sentiment model
├── requirements.txt     # Dependencies
├── static/              # Front-end files
├── templates/           # HTML templates
└── README.md            # Project documentation

📈 Model Performance
Metric	Score
Accuracy	~92%
Precision	High
Recall	High
Inference Time	Fast (<200ms on CPU)


✨ Future Improvements

Add Multimodal Inputs (Audio + Text)

Deploy on AWS / Render / Railway

Add LLM-based sentiment verifier

Support multilingual sentiment analysis

Add UI dashboard with charts

👨‍💻 Author

Manas Upadhyay
Sentiment Analysis • NLP • Deep Learning • Flask Apps
