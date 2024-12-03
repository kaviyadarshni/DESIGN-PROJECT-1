Step 1: Open Google Colab
•	Go to [Google Colab](https://colab.research.google.com/).
•	Sign in with your Google account if prompted.
Step 2: Create a New Notebook
•	Click on **"File"** > **"New Notebook"** to create a new notebook.
Step 3: Install Required Libraries
•	In the first cell, you can ensure all libraries are installed 
Step 4: Upload Your Dataset
•	Click on the folder icon on the left sidebar to open the file explorer.
•	Click on the upload icon (a paperclip) to upload your `Crop_recommendation.csv` file from your local machine.
Step 5: Write the Code
In a new cell, write the code
Import Libraries
import numpy as np
import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
#Suppress Warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
#Load the Dataset
df = pd.read_csv('Crop_recommendation.csv')
# Explore the Data
print(df.head())
print(df.describe())
Step 6: Data Visualization
•	Visualize Missing Values
 		sns.heatmap(df.isnull(), cmap="coolwarm")
   		plt.show()

•	Plot Distributions
   		plt.figure(figsize=(12, 5))
   		plt.subplot(1, 2, 1)
 		sns.distplot(df['temperature'], color="purple", bins=15, hist_kws={'alpha': 0.2})
  		plt.subplot(1, 2, 2)
   		sns.distplot(df['ph'], color="green", bins=15, hist_kws={'alpha': 0.2})
  		 plt.show()
•	Countplot and Pairplot
  sns.countplot(y='label', data=df, palette="plasma_r")
  plt.show()
  sns.pairplot(df, hue='label')
  plt.show()

Step 7: Data Preparation
•	Prepare Features and Labels
   		c = df.label.astype('category')
  		targets = dict(enumerate(c.cat.categories))
   		df['target'] = c.cat.codes
   		y = df.target
   		X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

•	Visualize Correlations
   		sns.heatmap(X.corr())
   		plt.show()
   Step 8: Model Training
•	Split the Data
   		from sklearn.model_selection import train_test_split
   		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
•	Scale the Data
   		 from sklearn.preprocessing import MinMaxScaler
   		 scaler = MinMaxScaler()
   		X_train_scaled = scaler.fit_transform(X_train)
   		X_test_scaled = scaler.transform(X_test)
•	Train the KNN Model
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier()
  knn.fit(X_train_scaled, y_train)
  print(knn.score(X_test_scaled, y_test))
Step 9: Make Predictions
•	Define the Prediction Function
   def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
       input_data = pd.DataFrame({
           'N': [N],
           'P': [P],
           'K': [K],
           'temperature': [temperature],
           'humidity': [humidity],
           'ph': [ph],
           'rainfall': [rainfall]
       })
       input_data_scaled = scaler.transform(input_data)
       prediction = knn.predict(input_data_scaled)[0]
       crop_name = targets[prediction]
       return crop_name
•	Example Usage
   		       predicted_crop = predict_crop(N=9, P=42, K=43, 
temperature=20.879744, humidity=82.002744, 
ph=6.502985, rainfall=202.935536)
  	                    print(f"The predicted best crop is: {predicted_crop}") 
Step 10: Run the Code
Execute each cell by clicking the play button on the left side of each cell. Make sure to run cells in order to avoid errors.
