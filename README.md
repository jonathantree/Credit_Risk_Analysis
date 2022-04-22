# Credit Risk Analysis
## Project Overview
This project aims to test various supervised classifier ML models on predicting credit risk for loan applications. Being an inherently imbalanced problem were the high- risk class is strongly outnumbered by the low-risk class, this project tests different resampling techniques in attempt to resolve the class imbalance. The libraries used in this project were `imbalanced-learn` and `scikit-learn`. Using the credit card credit dataset from LendingClub (LoanStats_2019Q1.csv.zip), the data was oversampled using the `RandomOverSampler` and `SMOTE` algorithms. An undersampling approach was also applied to the data using the `ClusterCentroids` algorithm. A combinatorial approach of over and undersampling using the `SMOTEENN` algorithm was used. The `BalancedRandomForestClassifier` and `EasyEnsembleClassifier` were also used to reduce bias in resampling were assessed. The performance metrics for these models were calculated and are reported in Table 1.

## Results

### **Table 1. Summary of Model Performance**
| **Resampling Technique/Model**  | **Balanced Accuracy Score** | **High-risk Precision** | **High-risk Recall** | **High-risk F1 Score** |
|---------------------------------|:-----------------------------:|:-------------------------:|:----------------------:|:------------------------:|
| RandomOverSampler Log. Reg.     | 0.67                        | 0.01                    | 0.71                 | 0.02                   |
| SMOTE Log. Reg.                 | 0.66                        | 0.01                    | 0.63                 | 0.02                   |
| ClusterCentroids Log. Reg.      | 0.54                        | 0.01                    | 0.69                 | 0.01                   |
| SMOTENN Log. Reg.               | 0.64                        | 0.01                    | 0.73                 | 0.02                   |
| BalancedRandomForestClassifier  | 0.79                        | 0.03                    | 0.70                 | 0.06                   |
| EasyEnsembleClassifier          | 0.93                        | 0.09                    | 0.92                 | 0.16                   |

## Summary
Of the six resampling techniques and models, the cluster centroid under-sampling technique was the lowest performing. With an accuracy score of 0.54 and a high-risk precision of 0.01. The cluster centroid low risk F1 score was also the lowest of the six approaches (0.57). The best performing model was the `EasyEnsembleClassifier`. This model obtained a high accuracy score of 0.93, and predicted high risk credit at a much higher rate than the other five techniques (0.09).  This model obtained a low-risk F1 score of 0.97. Though this model outperformed the others, the precision in classifying the high risk class is still quite low. The confusion matrix for this model is shown below. Out of the 101 high risk loan applications in the test data, this model correctly identified 93 and incorrectly classified 8 of them as low-risk. This level of accuracy is quite good, thus out of the six models trained, the `EasyEnsembleClassifier` wins the recommendation. According the results from evaluating these models, when attempting to classify credit risk for loan applications, an adaptive boosting technique is recommended.

### Confusion Matrix for the `EasyEnsembleClassifier` model
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Risk</th>
      <th>Predicted Low-Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Risk</th>
      <td>93</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Actual Low-Risk</th>
      <td>983</td>
      <td>16121</td>
    </tr>
  </tbody>
</table>
</div>

## Code for these models:
# Split the Data into Training and Testing


```python
# Create our features
X = df_encoded.drop(columns=['loan_status_high_risk','loan_status_low_risk'])
# Create our target
y = df.loan_status
```


```python
# Check the balance of our target values
y.value_counts()
```




    low_risk     68470
    high_risk      347
    Name: loan_status, dtype: int64




```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
Counter(y_train)
```




    Counter({'low_risk': 51366, 'high_risk': 246})



# Oversampling

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Print the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

### Naive Random Oversampling


```python
# implement random oversampling
from imblearn.over_sampling import RandomOverSampler
# Resample the training data with the RandomOversampler
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

Counter(y_resampled)
```




    Counter({'low_risk': 51366, 'high_risk': 51366})




```python
from sklearn.linear_model import LogisticRegression
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# make predictions
y_pred = model.predict(X_test)
```


```python
from sklearn.metrics import balanced_accuracy_score
#Calculate the balanced accuracy score
balanced_accuracy_score(y_test, y_pred)
```




    0.6747822291583695




```python
# Display the confusion matrix
from sklearn.metrics import confusion_matrix
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Risk", "Actual Low-Risk"],
    columns=["Predicted High-Risk", "Predicted Low-Risk"]
)

# Displaying results
display(cm_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Risk</th>
      <th>Predicted Low-Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Risk</th>
      <td>72</td>
      <td>29</td>
    </tr>
    <tr>
      <th>Actual Low-Risk</th>
      <td>6214</td>
      <td>10890</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.71      0.64      0.02      0.67      0.46       101
       low_risk       1.00      0.64      0.71      0.78      0.67      0.45     17104
    
    avg / total       0.99      0.64      0.71      0.77      0.67      0.45     17205
    
    

### SMOTE Oversampling


```python
# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE(random_state=1, sampling_strategy='auto').fit_resample(
    X_train, y_train
)
Counter(y_resampled)
```




    Counter({'low_risk': 51366, 'high_risk': 51366})




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.6571029647398791




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Risk", "Actual Low-Risk"],
    columns=["Predicted High-Risk", "Predicted Low-Risk"]
)

# Displaying results
display(cm_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Risk</th>
      <th>Predicted Low-Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Risk</th>
      <td>64</td>
      <td>37</td>
    </tr>
    <tr>
      <th>Actual Low-Risk</th>
      <td>5464</td>
      <td>11640</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.63      0.68      0.02      0.66      0.43       101
       low_risk       1.00      0.68      0.63      0.81      0.66      0.43     17104
    
    avg / total       0.99      0.68      0.63      0.80      0.66      0.43     17205
    
    

# Undersampling

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Print the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests


```python
# Resample the data using the ClusterCentroids resampler
from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
```




    Counter({'high_risk': 246, 'low_risk': 246})




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.5447046721744204




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Risk", "Actual Low-Risk"],
    columns=["Predicted High-Risk", "Predicted Low-Risk"]
)

# Displaying results
display(cm_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Risk</th>
      <th>Predicted Low-Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Risk</th>
      <td>70</td>
      <td>31</td>
    </tr>
    <tr>
      <th>Actual Low-Risk</th>
      <td>10325</td>
      <td>6779</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.69      0.40      0.01      0.52      0.28       101
       low_risk       1.00      0.40      0.69      0.57      0.52      0.27     17104
    
    avg / total       0.99      0.40      0.69      0.56      0.52      0.27     17205
    
    

# Combination (Over and Under) Sampling

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Print the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

```python
# Resample the training data with SMOTEENN
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=1)
X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
Counter(y_resampled)
```




    Counter({'high_risk': 51361, 'low_risk': 46653})




```python
# Train the Logistic Regression model using the resampled data
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_resampled, y_resampled)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.6439909835230483




```python
# Display the confusion matrix
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Risk", "Actual Low-Risk"],
    columns=["Predicted High-Risk", "Predicted Low-Risk"]
)

# Displaying results
display(cm_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Risk</th>
      <th>Predicted Low-Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Risk</th>
      <td>74</td>
      <td>27</td>
    </tr>
    <tr>
      <th>Actual Low-Risk</th>
      <td>7606</td>
      <td>9498</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.01      0.73      0.56      0.02      0.64      0.41       101
       low_risk       1.00      0.56      0.73      0.71      0.64      0.40     17104
    
    avg / total       0.99      0.56      0.73      0.71      0.64      0.40     17205
    
    


```python

```
# Ensemble Learners

1. Train the model using the training data. 
2. Calculate the balanced accuracy score from sklearn.metrics.
3. Print the confusion matrix from sklearn.metrics.
4. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
5. For the Balanced Random Forest Classifier onely, print the feature importance sorted in descending order (most important feature to least important) along with the feature score

### Balanced Random Forest Classifier


```python
# Resample the training data with the BalancedRandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
brf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=1) 
brf_model.fit(X_train,y_train)
```




    BalancedRandomForestClassifier(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = brf_model.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.7885466545953005




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Risk", "Actual Low-Risk"],
    columns=["Predicted High-Risk", "Predicted Low-Risk"]
)

# Displaying results
display(cm_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Risk</th>
      <th>Predicted Low-Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Risk</th>
      <td>71</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Actual Low-Risk</th>
      <td>2153</td>
      <td>14951</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.03      0.70      0.87      0.06      0.78      0.60       101
       low_risk       1.00      0.87      0.70      0.93      0.78      0.62     17104
    
    avg / total       0.99      0.87      0.70      0.93      0.78      0.62     17205
    
### Easy Ensemble AdaBoost Classifier


```python
# Train the EasyEnsembleClassifier
from imblearn.ensemble import EasyEnsembleClassifier 
eec = EasyEnsembleClassifier(n_estimators=100, random_state=1)
eec.fit(X_train,y_train)
```




    EasyEnsembleClassifier(n_estimators=100, random_state=1)




```python
# Calculated the balanced accuracy score
y_pred = eec.predict(X_test)
balanced_accuracy_score(y_test, y_pred)
```




    0.9316600714093861




```python
# Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm, index=["Actual High-Risk", "Actual Low-Risk"],
    columns=["Predicted High-Risk", "Predicted Low-Risk"]
)

# Displaying results
display(cm_df)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High-Risk</th>
      <th>Predicted Low-Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Risk</th>
      <td>93</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Actual Low-Risk</th>
      <td>983</td>
      <td>16121</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.09      0.92      0.94      0.16      0.93      0.87       101
       low_risk       1.00      0.94      0.92      0.97      0.93      0.87     17104
    
    avg / total       0.99      0.94      0.92      0.97      0.93      0.87     17205
    
    


```python

```
