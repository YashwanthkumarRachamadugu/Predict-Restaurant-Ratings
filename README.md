# 🍽️ Predict Restaurant Ratings

This project builds and deploys a machine learning model that predicts restaurant ratings based on various input features such as cuisines, services, location, and other restaurant-related attributes.

It provides both:

* A backend model training pipeline (`train_model.py`)
* A frontend Flask web app (`app.py`) for user-friendly prediction interface.

---

## 📌 Project Highlights

* ✅ Data preprocessing using Label Encoding and MultiLabel Binarizer
* ✅ Trained and saved model using scikit-learn
* ✅ Flask web application for live user interaction
* ✅ Pre-trained files for immediate deployment
* ✅ Ready-to-use CSV dataset

---

## 📂 Folder Structure

```
task 1 - predict restaurant ratings/
│
├── app.py                      # Flask app for prediction
├── train_model.py              # Script to train and save ML model
├── Dataset .csv                # Raw dataset
├── model.pkl                   # Trained machine learning model
├── label_encoder.pkl           # LabelEncoder for categorical columns
├── mlb.pkl                     # MultiLabelBinarizer for multi-valued columns
├── dataset_columns.pkl         # Column info for preprocessed inputs
├── Predict Restaurant Ratings.mov # Demo video
├── venv/                       # (Optional) Python virtual environment
```

---

## 🔧 Setup Instructions

### 🛠 Requirements

* Python 3.7+
* pip (Python package installer)
* (Optional) virtualenv for isolation

### 📦 Installation & Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/restaurant-rating-predictor.git
   cd restaurant-rating-predictor
   ```

2. **Create a Virtual Environment (Optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install Required Libraries**

   ```bash
   pip install -r requirements.txt
   ```

   If you don’t have `requirements.txt`, install manually:

   ```bash
   pip install flask pandas scikit-learn numpy
   ```

---

## ⚙️ Run the Application

Once everything is installed, start the web server using:

```bash
streamlit run app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000/     #sometimes it automatically redirects
```

You’ll see a form where you can enter restaurant details and get the predicted rating.

---

## 🧠 Training the Model (Optional)

If you want to retrain the model on your own:

```bash
python train_model.py
```

This will:

* Load the dataset
* Preprocess data
* Train a classification model
* Save the model and encoders into `.pkl` files

---


## 📌 Notes

* Make sure the `.pkl` files are in the same directory as `app.py` for prediction to work.
* Do not commit your `venv/` folder or `.pkl` files to public repositories if sensitive.

---
