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

```

# Results Analysis

In the proposed method, we used:

a) Soil prediction and crop recommendation  
b) Plant leaf disease detection and fertilizer recommendation  
c) Crop yield prediction

## GUI for Proposed Method

![GUI for proposed Method](https://github.com/user-attachments/assets/93e83628-0503-4a6a-96f7-122ef7d51710)

## Upload the Dataset Related to Soil

![Upload the dataset related to soil](https://github.com/user-attachments/assets/e0c6c56b-5636-46c8-bba1-ff35bf9e28b5)


## After Uploading the Soil Dataset, Details are Shown

![After uploading the soil dataset, details are shown]![image](https://github.com/user-attachments/assets/93d7cc2a-b010-4fdc-be39-c55f7bcc3f43)

## Confusion Matrix for Soil Classification and Plant Recommendation

![Confusion matrix for soil classification and plant recommendation](path/to/soil_confusion_matrix.png)

## Select Soil Image for Testing

![Select soil image for testing](path/to/select_soil_image.png)

## Soil Predicted and Crop Recommended

![Soil predicted and crop recommended](path/to/soil_crop_recommended.png)

## Uploading Leaf Disease Dataset

![Uploading leaf disease dataset](path/to/upload_leaf_disease_dataset.png)

## After Uploading, Details are Shown

![After uploading, details are shown](path/to/leaf_disease_details_shown.png)

## Confusion Matrix for Plant Disease Detection and Fertilizer Recommendation

![Confusion Matrix for plant disease detection and fertilizer recommendation](path/to/disease_fertilizer_confusion_matrix.png)

## Normal Leaf Without Disease

![Normal leaf without disease](path/to/normal_leaf_without_disease.png)

## Disease is Predicted and Fertilizer is Recommended

![Disease is predicted and fertilizer is recommended](path/to/disease_predicted_fertilizer_recommended.png)

## Linear Regression R² Score is Shown

![Linear regression R² score is shown](path/to/linear_regression_r2_score.png)

## Crop Yield Prediction Interface

![Crop yield prediction interface](path/to/crop_yield_prediction_interface.png)

