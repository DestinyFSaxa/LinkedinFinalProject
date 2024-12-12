#!/usr/bin/env python
# coding: utf-8

# Name: Destiny Floyd-McGenuiss - Final Project - OPAN 6607-200 Programming II: DATA INFRASTRUCTURE

# Part 1

# Q1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[84]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
file_path = '/Users/destinyfloyd/Desktop/Programming-II/social_media_usage.csv'
s = pd.read_csv(file_path)
s.shape


# Q2. Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[26]:


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


toydata = {'Income': [1, 2, 0], 'Gender': [1, 1, 0]}
df = pd.DataFrame(toydata)

print(clean_sm(df['Income']))
print(clean_sm(df['Gender']))


# Q3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable 
# (that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses 
# LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 
# to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing).
# Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[48]:


ss = pd.DataFrame()
ss['sm_li'] = s['web1h'].apply(clean_sm)
ss['income'] = s['income'].where(s['income'] <= 9, np.nan)
ss['educ2'] = s['educ2'].where(s['educ2'] <= 8, np.nan)
ss['marital'] = s['marital'].apply(clean_sm)
ss['gender'] = s['gender'].apply(clean_sm)
ss['par'] = s['par'].apply(clean_sm)
ss['age'] = s['age'].apply(lambda x: x if x <= 98 else np.nan)
ss.dropna(inplace=True)


print(ss.describe())
print(ss.corr())


# In[64]:


plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
sns.countplot(x='income', hue='sm_li', data=ss)
plt.title('Income vs LinkedIn Usage')

plt.subplot(2, 3, 2)
sns.countplot(x='educ2', hue='sm_li', data=ss)
plt.title('Education vs LinkedIn Usage')

plt.subplot(2, 3, 3)
sns.countplot(x='par', hue='sm_li', data=ss)
plt.title('Parenthood vs LinkedIn Usage')

plt.subplot(2, 3, 4)
sns.countplot(x='marital', hue='sm_li', data=ss)
plt.title('Marital Status vs LinkedIn Usage')

plt.subplot(2, 3, 5)
sns.countplot(x='gender', hue='sm_li', data=ss)
plt.title('Gender vs LinkedIn Usage')

plt.subplot(2, 3, 6)
sns.histplot(data=ss, x='age', hue='sm_li', multiple='stack', bins=20)
plt.title('Age Distribution vs LinkedIn Usage')

plt.tight_layout()
plt.show()


# Q4. Create a target vector (y) and feature set (X)

# In[66]:


y = ss['sm_li']

X = ss.drop(columns=['sm_li'])

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("Feature set X:")
print(X.head())
print("\nTarget vector y:")
print(y.head())


# Q5. Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Q6. Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[132]:


logistic_model = LogisticRegression(class_weight='balanced', random_state=42)

logistic_model.fit(X_train, y_train)

print("Model coefficients:", logistic_model.coef_)
print("Intercept:", logistic_model.intercept_)


# Q7. Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate
# a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[80]:


y_pred = logistic_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
cm_display.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()


# The accuracy score gives you a high-level view of the model's performance, while the confusion matrix offers a more granular understanding of how well the model performed on each class. When you analyze a confusion matrix, you can see model behavior and where the errors are happening in terms of False Positive or False Negative rates, which could then lead you to tune your model or use new features to mitigate any issues. The darkest shade of blue is decribing the true positive scores that we predicited to be positive which had a score of 98, the score of 63 are the false positive precicted, 24 is the false negative predicited, and 67 is the true negative predicted

# Q8. Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents.

# In[146]:


import pandas as pd

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(cm, 
                     index=['Actual Negative (Not Using LinkedIn)', 'Actual Positive (Using LinkedIn)'], 
                     columns=['Predicted Negative (Not Using LinkedIn)', 'Predicted Positive (Using LinkedIn)'])

print("Confusion Matrix:")
print(cm_df)


# Q9. Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# In[122]:


report = classification_report(y_test, y_pred, target_names=['Not Using LinkedIn', 'Using LinkedIn'], output_dict=True)

print("Classification Report:")
print(report)

precision_sklearn = report['Using LinkedIn']['precision']
recall_sklearn = report['Using LinkedIn']['recall']
f1_score_sklearn = report['Using LinkedIn']['f1-score']

print("\nManual vs Sklearn Metrics:")
print(f"Precision (manual): {precision:.4f}, Precision (sklearn): {precision_sklearn:.4f}")
print(f"Recall (manual): {recall:.4f}, Recall (sklearn): {recall_sklearn:.4f}")
print(f"F1 Score (manual): {fscore:.4f}, F1 Score (sklearn): {f1_score_sklearn:.4f}")


# Q10. Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[140]:


individual_1 = pd.DataFrame({
    'income': [8],
    'educ2': [7],
    'par': [0], 
    'marital': [0], 
    'gender': [1], 
    'age': [42]
})

individual_1 = individual_1[logistic_model.feature_names_in_]

probabilities_1 = logistic_model.predict_proba(individual_1)

print(f"Probability that individual 2 (age 84) uses LinkedIn: {probabilities_1[0][1]:.4f}")


# In[142]:


individual_2 = pd.DataFrame({
    'income': [8],
    'educ2': [7],
    'par': [0],  
    'marital': [1],  
    'gender': [0], 
    'age': [82]
})

individual_2 = individual_2[logistic_model.feature_names_in_]
probabilities_2 = logistic_model.predict_proba(individual_2)

print(f"Probability that individual 2 (age 82) uses LinkedIn: {probabilities_2[0][1]:.4f}")

