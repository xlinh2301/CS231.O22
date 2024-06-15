<h1 align="center">
Gender and Age Detection in Customer Analysis
</h1>

<table align="center">
  <tr>
    <th>Tên</th>
    <th>MSSV</th>
  </tr>
  <tr>
    <td>Nguyễn Xuân Linh</td>
    <td>22520775</td>
  </tr>
  <tr>
    <td>Lưu Khánh Vinh</td>
    <td>22521671</td>
  </tr>
  <tr>
    <td>Nguyễn Hữu Đức</td>
    <td>22520270</td>
  </tr>
  <tr>
    <td>Nguyễn Quốc Khánh</td>
    <td>22520646</td>
  </tr>
</table>
## Introduction
This project aims to determine the gender and age of customers using facial recognition to improve marketing strategies at Go! Dĩ An supermarket. By automating the collection of this data, we can help marketing departments tailor their campaigns more effectively, saving time and resources.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Preprocessing](#preprocessing)
4. [Algorithms and Models](#algorithms-and-models)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [References](#references)
8. [Additional Information from Notebook](#additional-information-from-notebook)

## Project Overview
The primary goal of this project is to utilize images from supermarket cameras to predict the age and gender of customers. This data can then be used to target marketing efforts more precisely.

## Dataset
We used the UTKFace dataset, which contains over 20,000 images of faces annotated with age, gender, and ethnicity. For our project, we focused on a subset of 9,780 images from the dataset.

### Classes
- **Age**: Divided into 5 groups
  - 0-6 years
  - 7-19 years
  - 20-32 years
  - 33-55 years
  - 56 years and older
- **Gender**: 
  - Male
  - Female

## Preprocessing
1. **Image Resizing**: All images were resized to 128x128 pixels.
2. **Grayscale Conversion**: Converted images to grayscale.
3. **Edge Detection**: Used the Canny edge detection algorithm to extract features.
4. **Data Splitting**: Split data into training (80%) and testing (20%) sets.
5. **Feature Scaling and PCA**: Standardized features and reduced dimensionality using Principal Component Analysis (PCA).

## Algorithms and Models
We implemented and compared several machine learning algorithms:
1. **Support Vector Machine (SVM)**
2. **k-Nearest Neighbors (KNN)**
3. **Logistic Regression**
4. **Random Forest**

## Results
- **SVM**: Achieved the highest accuracy for both age (62%) and gender (77%) predictions.
- **KNN**: Provided reasonable accuracy but was outperformed by SVM.
- **Logistic Regression**: Showed competitive results for gender prediction.
- **Random Forest**: Effective but not as accurate as SVM for this dataset.

## Conclusion
SVM proved to be the most effective model for predicting both age and gender. Future work could involve enhancing data preprocessing, expanding the dataset, and optimizing model parameters to improve accuracy.

## Additional Information from Notebook

### Data Loading and Exploration
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data_path = 'path_to_dataset.csv'
data = pd.read_csv(data_path)

# Display the first few rows of the dataset
print(data.head())

# Visualize the age distribution
plt.hist(data['age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

## Data Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming 'features' contains the image data and 'labels' contains the target variables
features = data.drop(columns=['age', 'gender'])
labels = data[['age', 'gender']]

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Applying PCA
pca = PCA(n_components=50)  # Adjust n_components as needed
pca_features = pca.fit_transform(scaled_features)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=0.2, random_state=42)


## Model Building and Training
from sklearn.svm import SVC

# Initialize the SVM model
svm_model = SVC(kernel='linear')

# Train the model
svm_model.fit(X_train, y_train['gender'])

# Predict using the trained model
gender_predictions = svm_model.predict(X_test)

## Evaluation Metrics
from sklearn.metrics import accuracy_score, classification_report

# Evaluate the model
accuracy = accuracy_score(y_test['gender'], gender_predictions)
print(f'Gender Prediction Accuracy: {accuracy}')

# Detailed classification report
print(classification_report(y_test['gender'], gender_predictions))

### References
- Course slides and materials
- [OpenCV Canny Edge Detection](https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html)
- [DataCamp K-Nearest Neighbor Classification](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn)
- [Scikit-learn GridSearchCV Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- ChatGPT 3.5
