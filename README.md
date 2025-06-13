# Machine Learning Analysis of Student Depression Dataset  

## Overview  
This project analyzes a student depression dataset using supervised and unsupervised machine learning techniques. The goal is to identify patterns and build predictive models for depression risk factors among students.  

## Key Features  
- **Data Preprocessing**: Handles missing values, encodes categorical data, and scales features  
- **Exploratory Analysis**: Visualizations including distribution plots and correlation heatmaps  
- **Supervised Learning**:  
  - Support Vector Machines (SVM)  
  - K-Nearest Neighbors (KNN)  
  - Logistic Regression  
  - Random Forest  
- **Unsupervised Learning**: KMeans clustering with elbow method optimization  
- **Model Evaluation**: Accuracy, precision, recall, F1-score, MSE, RMSE, MAE metrics  
- **Visualization**: Confusion matrices and PCA-reduced cluster plots  

## Dataset  
The "Student Depression Dataset" contains:  
- Demographic information (age, gender)  
- Academic factors (CGPA, study satisfaction)  
- Lifestyle indicators (sleep duration, dietary habits)  
- Mental health status (depression binary classification)  

## How to Use  
1. **Prerequisites**:  
   - Python 3.x  
   - Jupyter Notebook  
   - Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn  

2. **Setup**:  
   ```bash  
   pip install -r requirements.txt  
   jupyter notebook  
   ```  
   Open `ML_Final_Project.ipynb`  

3. **Execution**:  
   - Run cells sequentially  
   - View visualizations inline  
   - Model metrics are printed after each evaluation  

## Code Structure  
```python  
# Data Loading  
data = pd.read_csv("Student Depression Dataset.csv")  

# Preprocessing  
data.fillna()  # Handle missing values  
LabelEncoder()  # Encode categorical features  

# Visualization  
sns.countplot()  # Depression distribution  
sns.heatmap()    # Feature correlations  

# Modeling  
train_test_split()  
StandardScaler()  
model.fit()      # For each classifier  

# Evaluation  
accuracy_score()  
ConfusionMatrixDisplay()  

# Clustering  
KMeans()  
PCA()           # Dimensionality reduction  
```  

## Results  
| Model                | Accuracy | Precision | Recall | F1-Score |  
|----------------------|----------|-----------|--------|----------|  
| Support Vector Machine | 0.84     | 0.85      | 0.89   | 0.87     |  
| K-Nearest Neighbors  | 0.82     | 0.83      | 0.87   | 0.85     |  
| Logistic Regression  | 0.85     | 0.86      | 0.89   | 0.87     |  
| Random Forest        | 0.84     | 0.85      | 0.88   | 0.87     |  

**Clustering**: Optimal k=3 determined via elbow method  

## Future Work  
- Hyperparameter tuning with GridSearchCV  
- Address class imbalance using SMOTE  
- Deploy as a web application for real-time predictions  

Developed as a final project for Machine Learning course.  

--- 
