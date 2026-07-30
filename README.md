# 🕵️‍♂️ Instagram Fake Profile Detector

A machine learning web app built with **Streamlit** that predicts whether an Instagram account is **real or fake**, based on public profile features such as bio length, username structure, follower/following ratio, and more.

## ✨ Features

- Clean, dark-themed UI with a custom Instagram-inspired background
- Simple form to enter account details (name, username, bio, stats, settings)
- Real-time prediction using a trained **Random Forest** classifier
- Model confidence score displayed alongside the result

## 🧠 How it works

The app extracts several behavioral features from the input (e.g. ratio of digits in the username, whether the full name matches the username, description length, etc.), scales them with a pre-trained `StandardScaler`, and feeds them into a Random Forest model trained on labeled real/fake Instagram profile data.

## 🚀 Getting started

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
streamlit run Instagram_fake_profile.py
```

Make sure `scalar.pkl`, `model_forest.pkl`, and `bg.png` are in the same folder as `Instagram_fake_profile.py`.

## 🛠️ Tech stack

- Python
- Streamlit
- scikit-learn
- NumPy

## 📸 Preview

*(Add a screenshot of the app here)*

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
