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
├── datasets/
│   ├── PlantDiseaseDataset/         # Dataset for plant disease detection
│   ├── SoilDataset/                 # Dataset for soil prediction
│   ├── testImages/                  # Test images for model validation
│   ├── testsoil/                    # Test soil data for model validation
│   ├── Agricultural_yield.csv       # Agricultural yield dataset
├── models/
│   ├── model/                       # Pre-trained models or model definitions
│   ├── crop_main.py                 # Main script for crop prediction model
│   ├── testtrain.py                 # Script to test the training of models
├── results/
│   ├── results.docx                 # Document with model results
│   ├── Optimizing Agriculture PAPER.docx  # Paper on optimizing agriculture
│   ├── Optimizing Agriculture Using Machine Learning Techniques (1).docx  # ML techniques for agriculture
├── gui/
│   ├── gui_code.py                  # Code for the graphical user interface
│   ├── front.png                    # Front-end image for the interface
├── dependencies/
│   ├── packages_required.txt        # Python dependencies file
├── app/
│   ├── run.bat                      # Batch file to run the application
│   ├── README.md                    # Project documentation
