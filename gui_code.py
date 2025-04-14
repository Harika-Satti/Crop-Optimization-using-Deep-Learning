import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Load the dataset
data = pd.read_csv('Agricultural_yield.csv')

# Drop rows with missing production values
data.dropna(subset=['Production'], inplace=True)

# Encode categorical variables
label_encoder_state = LabelEncoder()
label_encoder_district = LabelEncoder()
label_encoder_crop = LabelEncoder()
label_encoder_season = LabelEncoder()

data['State'] = label_encoder_state.fit_transform(data['State_Name'])
data['District'] = label_encoder_district.fit_transform(data['District_Name'])
data['Crop'] = label_encoder_crop.fit_transform(data['Crop'])
data['Season'] = label_encoder_season.fit_transform(data['Season'])

# Remove non-numeric values from target variable
data_numeric = data.copy()  # Make a copy of the dataframe
data_numeric['Production'] = pd.to_numeric(data_numeric['Production'], errors='coerce')
data_numeric.dropna(subset=['Production'], inplace=True)

# Split data into features and target variable
X = data_numeric[['State', 'District', 'Crop', 'Season', 'Area', 'Crop_Year']]
y = data_numeric['Production']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Linear Regression Model
model = LinearRegression()  # Linear Regression Regressor
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print("R-squared (R2) Score:", r2)

# GUI
def create_gui():
    def get_prediction():
        state = state_var.get()
        district = district_var.get()
        crop = crop_var.get()
        season = season_var.get()
        area = float(area_entry.get())
        crop_year = int(crop_year_entry.get())

        try:
            # Encode the inputs
            state_encoded = label_encoder_state.transform([state])[0]
            district_encoded = label_encoder_district.transform([district])[0]
            crop_encoded = label_encoder_crop.transform([crop])[0]
            season_encoded = label_encoder_season.transform([season])[0]

            # Predict
            user_data = [[state_encoded, district_encoded, crop_encoded, season_encoded, area, crop_year]]
            predicted_production = model.predict(user_data)

            result_label.config(text=f"Predicted Production: {predicted_production[0]:.2f}")
        except ValueError as e:
            messagebox.showerror("Error", f"Unseen label encountered: {str(e)}")

    root = tk.Tk()
    root.title("Crop Production Prediction")

    # Predefined options for State, District, Crop, and Season
    states = label_encoder_state.classes_
    districts = label_encoder_district.classes_
    crops = label_encoder_crop.classes_
    seasons = label_encoder_season.classes_

    # State Dropdown
    tk.Label(root, text="State Name:").grid(row=0, column=0)
    state_var = tk.StringVar(root)
    state_var.set(states[0])  # default value
    state_menu = tk.OptionMenu(root, state_var, *states)
    state_menu.grid(row=0, column=1)

    # District Dropdown
    tk.Label(root, text="District Name:").grid(row=1, column=0)
    district_var = tk.StringVar(root)
    district_var.set(districts[0])  # default value
    district_menu = tk.OptionMenu(root, district_var, *districts)
    district_menu.grid(row=1, column=1)

    # Crop Dropdown
    tk.Label(root, text="Crop:").grid(row=2, column=0)
    crop_var = tk.StringVar(root)
    crop_var.set(crops[0])  # default value
    crop_menu = tk.OptionMenu(root, crop_var, *crops)
    crop_menu.grid(row=2, column=1)

    # Season Dropdown
    tk.Label(root, text="Season:").grid(row=3, column=0)
    season_var = tk.StringVar(root)
    season_var.set(seasons[0])  # default value
    season_menu = tk.OptionMenu(root, season_var, *seasons)
    season_menu.grid(row=3, column=1)

    # Area and Crop Year entries
    tk.Label(root, text="Area:").grid(row=4, column=0)
    area_entry = tk.Entry(root)
    area_entry.grid(row=4, column=1)

    tk.Label(root, text="Crop Year:").grid(row=5, column=0)
    crop_year_entry = tk.Entry(root)
    crop_year_entry.grid(row=5, column=1)

    # Predict Button
    predict_button = tk.Button(root, text="Predict", command=get_prediction)
    predict_button.grid(row=6, column=1)

    # Result Label
    result_label = tk.Label(root, text="Predicted Production:")
    result_label.grid(row=7, column=1)

    root.mainloop()

# Run the GUI
create_gui()
