import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------------------
# Dataset class (title + review)
# ------------------------------
class ReviewDataset(Dataset):
    def __init__(self, titles, reviews, labels, tokenizer, max_len=128):
        self.titles = titles
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        title = str(self.titles[idx])
        review = str(self.reviews[idx])

        title_encoding = self.tokenizer(
            title,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        review_encoding = self.tokenizer(
            review,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        input_ids_title = title_encoding['input_ids'].squeeze(0)
        attention_mask_title = title_encoding['attention_mask'].squeeze(0)
        input_ids_review = review_encoding['input_ids'].squeeze(0)
        attention_mask_review = review_encoding['attention_mask'].squeeze(0)

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return input_ids_title, attention_mask_title, input_ids_review, attention_mask_review, label

# ------------------------------
# Model class (title + review)
# ------------------------------
class MultiModalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.title_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.review_bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768*2, 2)  # 2 classes: Positive/Negative

    def forward(self, input_ids_title, attention_mask_title, input_ids_review, attention_mask_review):
        title_output = self.title_bert(input_ids=input_ids_title, attention_mask=attention_mask_title)
        review_output = self.review_bert(input_ids=input_ids_review, attention_mask=attention_mask_review)

        title_cls = title_output.last_hidden_state[:,0,:]
        review_cls = review_output.last_hidden_state[:,0,:]

        combined = torch.cat((title_cls, review_cls), dim=1)
        x = self.dropout(combined)
        logits = self.classifier(x)
        return logits

# ------------------------------
# Load CSV (no header)
# ------------------------------
df = pd.read_csv("train.csv", header=None, names=['label', 'title', 'review'])

# Convert labels: 1 → 0, 2 → 1
df['label'] = df['label'].astype(int) - 1

# Optional: sanity check
print("Label distribution:", df['label'].value_counts())

titles = df['title'].tolist()
reviews = df['review'].tolist()
labels = df['label'].tolist()

# ------------------------------
# Split dataset
# ------------------------------
train_titles, val_titles, train_reviews, val_reviews, train_labels, val_labels = train_test_split(
    titles, reviews, labels, test_size=0.1, random_state=42
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_dataset = ReviewDataset(train_titles, train_reviews, train_labels, tokenizer)
val_dataset = ReviewDataset(val_titles, val_reviews, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# ------------------------------
# Training setup
# ------------------------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = MultiModalSentimentModel().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# ------------------------------
# Training loop
# ------------------------------
epochs = 3
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for input_ids_title, attention_mask_title, input_ids_review, attention_mask_review, labels_batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids_title = input_ids_title.to(device)
        attention_mask_title = attention_mask_title.to(device)
        input_ids_review = input_ids_review.to(device)
        attention_mask_review = attention_mask_review.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()
        logits = model(input_ids_title, attention_mask_title, input_ids_review, attention_mask_review)
        loss = criterion(logits, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1} Training Loss: {train_loss / len(train_loader):.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids_title, attention_mask_title, input_ids_review, attention_mask_review, labels_batch in val_loader:
            input_ids_title = input_ids_title.to(device)
            attention_mask_title = attention_mask_title.to(device)
            input_ids_review = input_ids_review.to(device)
            attention_mask_review = attention_mask_review.to(device)
            labels_batch = labels_batch.to(device)

            logits = model(input_ids_title, attention_mask_title, input_ids_review, attention_mask_review)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels_batch).sum().item()
            total += labels_batch.size(0)
    print(f"Epoch {epoch+1} Validation Accuracy: {correct/total:.4f}")

# ------------------------------
# Save trained model
# ------------------------------
torch.save(model.state_dict(), "multimodal_sentiment_model.pt")
print("Model saved as multimodal_sentiment_model.pt")
