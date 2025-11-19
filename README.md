# 🧵 Day 22 — Fashion Item Classifier (Neural Networks)

This project builds a **Convolutional Neural Network (CNN)** to classify **Fashion-MNIST** images  
into 10 clothing categories (T-shirt, Trouser, Dress, Coat, Sneakers, etc.).

It demonstrates how deep learning models perform **image recognition**, similar to systems used in  
retail, fashion-tech, and e-commerce catalog automation.

---

## 🚀 Overview

- Loads the **Fashion-MNIST dataset** (70,000 grayscale images, 28×28)
- Normalizes and reshapes data for CNN input
- Builds a **lightweight CNN architecture**
- Trains for **6 epochs** with validation
- Generates:
  - 📌 `fashion_cnn.h5` — trained model  
  - 📌 `training_history.png` — training vs validation accuracy  
  - 📌 `confusion_matrix.png` — visual evaluation  
  - 📌 `sample_predictions.png` — random model predictions  
- Produces a **classification report** and **confusion matrix** using scikit-learn

---

## 🧠 Workflow

1. **Data Preprocessing**  
   Loads Fashion-MNIST → scales pixel values → reshapes to `(28, 28, 1)`.

2. **Model Architecture (CNN)**  
   - Conv2D → MaxPool  
   - Conv2D → MaxPool  
   - Dense + Dropout  
   - Output Softmax layer (10 classes)

3. **Training**  
   Adam optimizer + sparse categorical crossentropy.

4. **Evaluation**  
   - Accuracy & loss  
   - Classification report  
   - Confusion matrix  

5. **Visualization**  
   Saves training curves and sample predictions as images.

---

## 📊 Example Output (Demo)

### ⭐ Sample Predictions (Random Images)
- T-shirt → predicted: Shirt  
- Sneaker → predicted: Sneaker  
- Dress → predicted: Dress  
(12 images displayed in generated PNG)

### ⭐ Test Accuracy
Typical accuracy from this CNN: **≈ 89–91%**

---

## 🧩 Tech Stack

Python | TensorFlow | Keras | NumPy | Pandas | Matplotlib | Seaborn | Scikit-learn

---

## ▶️ Running the Project

```bash
source ../Day-01-Titanic/venv/bin/activate
pip install -r requirements.txt
python3 run_fashion_cnn.py
```

Outputs:
- `fashion_cnn.h5`
- `training_history.png`
- `confusion_matrix.png`
- `sample_predictions.png`

---

## 🔗 Connect

LinkedIn: https://www.linkedin.com/in/abhineet-s  
GitHub: (add repo link once pushed)
