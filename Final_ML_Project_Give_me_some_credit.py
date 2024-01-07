#!/usr/bin/env python
# coding: utf-8

# # Initial Data Exploration

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#Import necessary libraries for data analysis and visualization
#This cell imports the essential Python libraries for data manipulation and visualization. 
#Pandas is used for data manipulation and analysis, NumPy for numerical operations,
#Seaborn and Matplotlib for plotting and visualizing data.


# In[3]:


train = pd.read_csv('/Users/ranaibrahimli/Documents/ML_Project/assets/cs-training.csv')
test = pd.read_csv('/Users/ranaibrahimli/Documents/ML_Project/assets/cs-test.csv')


# In[4]:


train.shape


# In[5]:


test.shape


# In[6]:


train.head()


# # Data Cleaning and Preprocessing

# In[7]:


#Checking and removing any duplicates in the data

train.duplicated().value_counts()


# In[8]:


#Let's delete them

train_redup = train.drop_duplicates()


# In[9]:


train_redup.duplicated().sum()


# In[10]:


#Checking the null values
#Calculate the percentage of missing values for each column in the training data

round(train_redup.isnull().sum()/train_redup.shape[0]*100, 2)


# In[11]:


#Define a function to find missing values
def findMiss(df):
    return round(df.isnull().sum()/df.shape[0]*100, 2)


# In[104]:


#Verify the shape of a potentially reduced dataset
train_redup.shape


# In[13]:


#Use the findMiss function on the reduced dataset
findMiss(train_redup)


# # Data Analysis and Visualization

# In[14]:


#Finding the missing values

train_redup[train_redup.NumberOfDependents.isnull()].describe()


# In[15]:


train_redup.NumberOfDependents.max()


# In[16]:


train_redup['NumberOfDependents'].agg(['mode'])


# In[17]:


train_redup.groupby(['NumberOfDependents']).size()


# In[18]:


sns.countplot(x='SeriousDlqin2yrs', data=train_redup)
plt.show()


# In[19]:


sns.boxplot(x='SeriousDlqin2yrs', y='DebtRatio', data=train_redup)
plt.show()


# In[23]:


#This section separates the dataset into two parts based on missing values in the 'NumberOfDependents' column.
fam_miss = train_redup[train_redup.NumberOfDependents.isnull()]
fam_nmiss = train_redup[train_redup.NumberOfDependents.notnull()]


# In[24]:


fam_miss.shape


# In[25]:


fam_nmiss.shape


# In[26]:


#Imputing Missing Data
fam_miss['NumberOfDependents'] = fam_miss['NumberOfDependents'].fillna(0)
fam_miss['MonthlyIncome'] = fam_miss['MonthlyIncome'].fillna(0)


# In[27]:


findMiss(fam_miss)


# In[28]:


findMiss(fam_nmiss)


# In[29]:


#Aggregating Statistical Measures for Income
fam_nmiss['MonthlyIncome'].agg(['mean','median','min'])


# In[30]:


fam_nmiss['MonthlyIncome'].agg(['max'])


# In[31]:


fam_nmiss['MonthlyIncome'] = fam_nmiss['MonthlyIncome'].fillna(fam_nmiss['MonthlyIncome'].median())


# In[32]:


#Finalizing Data Imputation
findMiss(fam_nmiss)


# In[33]:


filled_train = pd.concat([fam_nmiss,fam_miss])


# In[34]:


#Concatenating Data Subsets
findMiss(filled_train)


# In[35]:


filled_train.head()


# In[36]:


#Analyzing Feature Distributions and Outliers
filled_train.groupby(['SeriousDlqin2yrs']).size()/filled_train.shape[0]


# In[37]:


filled_train.RevolvingUtilizationOfUnsecuredLines.describe()


# In[38]:


filled_train['RevolvingUtilizationOfUnsecuredLines'].quantile([.99])


# In[39]:


filled_train[filled_train['RevolvingUtilizationOfUnsecuredLines'] > 10].describe()


# In[40]:


#Dropping Outliers Based on Revolving Utilization
util_droped = filled_train.drop(filled_train[filled_train['RevolvingUtilizationOfUnsecuredLines'] > 10].index)


# In[41]:


#Visualizing Age Distribution After Outlier Removal
#The majority of the data points fall within the box, suggesting a consistent age range among most of the borrowers, with only a few borrowers being significantly younger or older than the rest.
sns.boxplot(util_droped['age'])


# In[42]:


util_droped.groupby(['NumberOfTime30-59DaysPastDueNotWorse']).size()


# In[43]:


util_droped.groupby(['NumberOfTime60-89DaysPastDueNotWorse']).size()


# In[44]:


util_droped.groupby(['NumberOfTimes90DaysLate']).size()


# In[45]:


#Analyzing Delinquency in Relation to Serious Delinquency
util_droped[util_droped['NumberOfTimes90DaysLate']>=96].head()


# In[46]:


util_droped[util_droped['NumberOfTimes90DaysLate']>=96]['SeriousDlqin2yrs'].describe()


# In[47]:


#Grouping by Serious Delinquency After Removing Extreme Cases of Late Payments

util_droped[util_droped['NumberOfTimes90DaysLate']>=96].groupby(['SeriousDlqin2yrs']).size()


# In[48]:


util_droped.head()


# In[49]:


util_droped['DebtRatio'].describe()


# In[50]:


sns.kdeplot(util_droped['DebtRatio'])


# In[51]:


#Calculating the 97.5th percentile of the 'DebtRatio' variable, which is a common technique to identify and potentially remove outliers.
util_droped['DebtRatio'].quantile([.975])


# In[52]:


util_droped[util_droped['DebtRatio']>3492][['SeriousDlqin2yrs','MonthlyIncome']].describe()


# In[53]:


temp = util_droped[(util_droped['DebtRatio']>3492) & (util_droped['SeriousDlqin2yrs']==util_droped['MonthlyIncome'])]


# In[54]:


#isolating specific cases and examining the size of each group concerning serious delinquency. 
temp.groupby(['SeriousDlqin2yrs']).size()


# In[55]:


#Removing the nondefaulters
dRatio = util_droped.drop(util_droped[(util_droped['DebtRatio']>3492) & (util_droped['SeriousDlqin2yrs']==util_droped['MonthlyIncome'])].index)


# In[56]:


dRatio.head()


# In[57]:


plt.hist(util_droped['DebtRatio'], bins=50, color='blue', edgecolor='black')
plt.title('DebtRatio Distribution')
plt.xlabel('DebtRatio')
plt.ylabel('Frequency')
plt.show()


# In[58]:


plt.scatter(util_droped['age'], util_droped['MonthlyIncome'])
plt.title('Age vs. MonthlyIncome')
plt.xlabel('Age')
plt.ylabel('MonthlyIncome')
plt.show()


# In[59]:


sns.boxplot(x='SeriousDlqin2yrs', y='MonthlyIncome', data=util_droped)
plt.title('MonthlyIncome Distribution by SeriousDlqin2yrs')
plt.xlabel('SeriousDlqin2yrs')
plt.ylabel('MonthlyIncome')
plt.show()


# In[60]:


sns.kdeplot(util_droped[util_droped['SeriousDlqin2yrs'] == 0]['MonthlyIncome'], label='Not SeriousDlqin2yrs', shade=True)
sns.kdeplot(util_droped[util_droped['SeriousDlqin2yrs'] == 1]['MonthlyIncome'], label='SeriousDlqin2yrs', shade=True)
plt.title('MonthlyIncome Distribution by SeriousDlqin2yrs Status')
plt.xlabel('MonthlyIncome')
plt.ylabel('Density')
plt.legend()
plt.show()


# In[61]:


corr = util_droped.corr()  


# In[62]:


plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 10})
plt.title('Correlation Heatmap')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[107]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler


# In[108]:


#Impute missing values before splitting
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)


# In[109]:


#Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)


# In[111]:


#Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[112]:


#Fit the model
model = LogisticRegression()
model.fit(X_train, y_train)


# In[113]:


#Make predictions
y_pred = model.predict(X_test)


# # Model Evaluation

# In[114]:


#Classification report
print(classification_report(y_test, y_pred))


# In[119]:


X = train_redup.drop('SeriousDlqin2yrs', axis=1)  # Features
y = train_redup['SeriousDlqin2yrs']  # Target variable


# In[120]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[121]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train) 
X_train_imputed = imputer.transform(X_train) 
X_test_imputed = imputer.transform(X_test)


# In[117]:


imputer = SimpleImputer(strategy='mean')


# In[123]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train_imputed)  # Fit the scaler on the imputed training data
X_train_scaled = scaler.transform(X_train_imputed)  # Scale the imputed training data
X_test_scaled = scaler.transform(X_test_imputed)  # Scale the imputed test data


# In[72]:


# Scale the features after imputation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)


# In[73]:


# Initialize the Logistic Regression model
logreg = LogisticRegression()


# In[74]:


# Fit the model on the scaled and imputed training data
logreg.fit(X_train_scaled, y_train)


# In[75]:


# Drop rows with missing values
X_train_dropped = X_train.dropna()
y_train_dropped = y_train.loc[X_train_dropped.index]


# In[77]:


# Predictions
y_pred = logreg.predict(X_test_scaled)
y_pred_proba = logreg.predict_proba(X_test_scaled)[:,1]


# In[78]:


# Print classification report
print(classification_report(y_test, y_pred))


# In[79]:


# Compute and print AUC score
print("AUC Score: ", roc_auc_score(y_test, y_pred_proba))


# # Feature Importance

# In[84]:


get_ipython().system('pip install graphviz')


# In[85]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn import tree


# In[86]:


dt_classifier = DecisionTreeClassifier(random_state=42)


# In[87]:


dt_classifier.fit(X_train, y_train)


# In[88]:


y_pred_dt = dt_classifier.predict(X_test)


# In[89]:


get_ipython().system('dot -V')


# In[90]:


from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import display


# In[91]:


from sklearn.tree import DecisionTreeClassifier


# In[92]:


dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=3)  # Set a small depth to make the tree smaller


# In[93]:


dt_classifier.fit(X_train, y_train)


# In[94]:


dot_data = export_graphviz(dt_classifier, out_file=None, feature_names=X_train.columns, 
                           class_names=['No Default', 'Default'], filled=True, 
                           rounded=True, special_characters=True)


# In[95]:


graph = Source(dot_data)


# In[96]:


display(graph)


# In[97]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


# In[98]:


imputer = SimpleImputer(strategy='mean')


# In[99]:


X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[100]:


rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100)


# In[101]:


rf_classifier.fit(X_train_imputed, y_train)


# In[102]:


predictions = rf_classifier.predict(X_test_imputed)


# In[103]:


feature_importances = pd.Series(rf_classifier.feature_importances_, index=X_train.columns)
feature_importances.nlargest(10).plot(kind='barh')  # You can adjust the number as needed
plt.title('Feature Importances in Random Forest')
plt.show()


# # Conclusion

# We can notice that LogisticRegression generally performed better that RandomForestClassifier
