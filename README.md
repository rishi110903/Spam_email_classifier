# 📧 Spam Email Classifier

This project uses **Logistic Regression** and **TF-IDF Vectorization** to classify messages as **Spam or Ham**.

## 🚀 Features
- Preprocessing using TF-IDF
- Logistic Regression model for classification
- 98% accuracy on test data
- Predicts whether a message is spam or not directly from CLI

## 🧠 Dataset
The dataset used is `spam_ham_dataset.csv`, which contains labeled spam and ham messages.

| Column | Description |
|---------|-------------|
| Message | The message content |
| Category | Label (spam = 0, ham = 1) |

## 🧩 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- NLTK

## 🖥️ How to Run
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/spam-detection-ML.git
cd spam-detection-ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the program
python main.py
