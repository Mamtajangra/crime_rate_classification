
# Crime Classification using Random Forest

This project focuses on predicting the type of crime (`OFFENSE_CODE_GROUP`) based on various features such as district, day of the week, and whether a shooting occurred, using the Random Forest Classifier from Scikit-learn.

---

##  Project Overview

- **Objective**: Build a machine learning model to classify types of crimes in Boston based on structured police incident data.
- **Dataset**: Boston crime data (`crime.csv`)
- **Algorithm Used**: Random Forest Classifier
- **Output**: Classification report showing model performance

---

##  Dataset Description

The dataset includes various features related to criminal incidents. Relevant columns used:

| Column Name         | Description |
|---------------------|-------------|
| `DISTRICT`          | Police district |
| `SHOOTING`          | Whether the incident involved a shooting (Yes/No) |
| `DAY_OF_WEEK`       | Day when the incident occurred |
| `UCR_PART`          | Crime classification type |
| `OFFENSE_CODE_GROUP` | **Target column** – Category of the offense |

---

##  Technologies Used

- Python 
- Pandas 
- NumPy
- Matplotlib & Seaborn 
- Scikit-learn 

---

##  Steps Followed

### 1. **Data Loading & Exploration**
```python
df = pd.read_csv("crime.csv", encoding="latin1")
```
- Used `.info()`, `.describe()`, `.head()`, `.tail()` to understand structure
- Checked null values using `.isnull().sum()`

### 2. **Data Cleaning**
```python
df = df.drop(columns = ["INCIDENT_NUMBER", "STREET", "Location", "OFFENSE_CODE"])
df['SHOOTING'] = df['SHOOTING'].fillna('0')
df['SHOOTING'] = df['SHOOTING'].replace({'Y': 1, '0': 0})
```

### 3. **Label Encoding**
```python
label_cols = ['DISTRICT', 'DAY_OF_WEEK', 'UCR_PART']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
```

### 4. **Feature Selection**
```python
X = df.drop(columns=["OFFENSE_CODE_GROUP"])
y = df["OFFENSE_CODE_GROUP"]
```

### 5. **Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 6. **Model Training**
```python
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
```

### 7. **Evaluation**
```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

##  Sample Output

```
              precision    recall  f1-score   support
     Larceny       0.90      0.88      0.89       540
   Vandalism     0.78      0.75      0.76       230
         ...     ...      ...      ...         ...
    Arson         0.65      0.70      0.67        95

    accuracy                           0.85      2000
   macro avg       0.81      0.80      0.80      2000
weighted avg       0.85      0.85      0.85      2000
```

*( actual output may vary depending on dataset size and splits.)*

---

##  Conclusion

This project demonstrates how to preprocess real-world crime data, apply encoding, train a Random Forest model, and evaluate classification performance. It can be extended to include more advanced NLP or geospatial crime analysis.

---

##  Future Work

- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Visualizing feature importances
- Integrating time series patterns for temporal crime trends
- Using other ML models for comparison

---


