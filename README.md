# Crop-Optimization-using-Deep-Learning

This project focuses on developing an integrated system for agricultural optimization, including soil prediction, crop recommendation, plant disease detection, fertilizer suggestion, and crop yield prediction. The system leverages machine learning and deep learning techniques to assist farmers in making informed decisions to enhance agricultural productivity and sustainability.

---

## ✨ Project Overview

The system analyzes soil images to classify soil types using a **Convolutional Neural Network (CNN)**.  
Based on the soil type, it recommends suitable crops using **Random Forest** and **XGBoost** algorithms.  
It also detects plant diseases using a CNN model based on the **MobileNet** architecture, and suggests fertilizers accordingly.  
Finally, it predicts crop yield using **LSTM (Long Short-Term Memory)** models by considering soil parameters and location details.

---

## 🚀 Features

- Soil Type Prediction
- Crop Recommendation
- Plant Disease Detection
- Fertilizer Suggestion
- Crop Yield Prediction
- Integrated Decision Support System

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Scikit-learn
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Tkinter (for GUI)

---

## 📈 Workflow

1. Upload soil images → **CNN model** predicts soil type.
2. Based on soil → **Random Forest/XGBoost** recommend suitable crops.
3. Upload plant leaf images → **MobileNet CNN** detects diseases.
4. Suggest fertilizers based on disease/soil analysis.
5. Use **LSTM** to predict crop yield based on location and parameters.

---

## 📂 Project Structure

```bash
├── soil_prediction/
│   ├── cnn_model.py
├── crop_recommendation/
│   ├── random_forest.py
│   ├── xgboost.py
├── plant_disease_detection/
│   ├── mobilenet_cnn.py
├── fertilizer_suggestion/
│   ├── fertilizer_recommender.py
├── crop_yield_prediction/
│   ├── lstm_model.py
├── dataset/
│   ├── soil_images/
│   ├── leaf_images/
├── app.py
├── requirements.txt
├── README.md
