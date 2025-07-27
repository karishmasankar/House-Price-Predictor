{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "39c703bf-17d4-4751-961f-b48558787dda",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "               Model        MAE       RMSE  R2 Score\n",
      "0  Linear Regression   80879.10  100444.06    0.9180\n",
      "1      Decision Tree  142942.57  181892.04    0.7311\n",
      "2      Random Forest   95337.40  121001.62    0.8810\n",
      "3  Gradient Boosting   87404.15  109446.49    0.9026\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv(r\"C:\\Users\\Karishma\\Desktop\\my AI project\\House_Price_Predictor\\data\\USA_Housing.csv\")\n",
    "\n",
    "# Features and target\n",
    "X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',\n",
    "        'Avg. Area Number of Bedrooms', 'Area Population']]\n",
    "y = df['Price']\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Models to compare\n",
    "models = {\n",
    "    \"Linear Regression\": LinearRegression(),\n",
    "    \"Decision Tree\": DecisionTreeRegressor(),\n",
    "    \"Random Forest\": RandomForestRegressor(n_estimators=100),\n",
    "    \"Gradient Boosting\": GradientBoostingRegressor()\n",
    "}\n",
    "\n",
    "# Compare models\n",
    "results = []\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "\n",
    "    mae = mean_absolute_error(y_test, y_pred)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "    results.append((name, round(mae, 2), round(rmse, 2), round(r2, 4)))\n",
    "\n",
    "# Display results\n",
    "results_df = pd.DataFrame(results, columns=[\"Model\", \"MAE\", \"RMSE\", \"R2 Score\"])\n",
    "print(results_df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c010ecb4-7aba-4ac9-802a-861853c10f99",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tkinter as tk\n",
    "from tkinter import messagebox\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model\n",
    "with open('house_price_model.pkl', 'rb') as file:\n",
    "    model = pickle.load(file)\n",
    "\n",
    "# Create GUI window\n",
    "window = tk.Tk()\n",
    "window.title(\"House Price Predictor\")\n",
    "window.geometry(\"400x400\")\n",
    "\n",
    "# List of features (change based on your actual features)\n",
    "features = [\"Avg. Area Income\", \"Avg. Area House Age\", \"Avg. Area Number of Rooms\", \"Avg. Area Number of Bedrooms\", \"Area Population\"]\n",
    "entries = []\n",
    "\n",
    "# Create input fields\n",
    "for i, feature in enumerate(features):\n",
    "    label = tk.Label(window, text=feature)\n",
    "    label.pack()\n",
    "    entry = tk.Entry(window)\n",
    "    entry.pack()\n",
    "    entries.append(entry)\n",
    "\n",
    "# Prediction function\n",
    "def predict():\n",
    "    try:\n",
    "        input_data = [float(entry.get()) for entry in entries]\n",
    "        input_array = np.array([input_data])\n",
    "        prediction = model.predict(input_array)[0]\n",
    "        messagebox.showinfo(\"Prediction\", f\"Estimated House Price: ${prediction:,.2f}\")\n",
    "    except Exception as e:\n",
    "        messagebox.showerror(\"Error\", f\"Invalid input. Please enter numeric values.\\n{e}\")\n",
    "\n",
    "# Predict button\n",
    "btn = tk.Button(window, text=\"Predict\", command=predict)\n",
    "btn.pack(pady=10)\n",
    "\n",
    "window.mainloop()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "1728b53a-ddc7-40c6-8707-aaea0b1aad44",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model saved successfully.\n"
     ]
    }
   ],
   "source": [
    "import joblib\n",
    "import os\n",
    "\n",
    "# Create 'models' folder if it doesn't exist\n",
    "os.makedirs('models', exist_ok=True)\n",
    "\n",
    "# Save the trained model to a file\n",
    "joblib.dump(LinearRegression(), 'models/house_price_model.pkl')\n",
    "\n",
    "print(\"✅ Model saved successfully.\")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8deb07ce-d91e-4d99-b4c2-7c57364a1016",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
