#!/usr/bin/env python
# coding: utf-8

# In[51]:


#Importing the Libraries
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#%matplotlib.pylot as plot
import seaborn as sns
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import imblearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score


# In[36]:


#Data Collection and Preparation
#Read The Data Set
df=pd.read_csv("E:\\NMDS\pl_train.csv")
df.head()


# In[37]:


df.tail()


# In[38]:


#Data Collection and Preparation
#Read The Data Set
df1=pd.read_csv("E:\\NMDS\pl_test.csv")
df1.head()


# In[39]:


df1.tail()


# In[40]:


df.info()


# In[41]:


df1.info()


# In[42]:


df.isnull().sum()


# In[43]:


df1.isnull().sum()


# In[44]:


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])


# In[45]:


df1['Gender']=df1['Gender'].fillna(df1['Gender'].mode()[0])
df1['Married']=df1['Married'].fillna(df1['Married'].mode()[0])


# In[46]:


#replacing + with space for filling the nan values
df['Dependents']=df['Dependents'].str.replace('+','')
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mode()[0])
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mode()[0])


# In[47]:


#replacing + with space for filling the nan values
df1['Dependents']=df1['Dependents'].str.replace('+','')
df1['Dependents']=df1['Dependents'].fillna(df1['Dependents'].mode()[0])
df1['Self_Employed']=df1['Self_Employed'].fillna(df1['Self_Employed'].mode()[0])
df1['LoanAmount']=df1['LoanAmount'].fillna(df1['LoanAmount'].mode()[0])
df1['Loan_Amount_Term']=df1['Loan_Amount_Term'].fillna(df1['Loan_Amount_Term'].mode()[0])
df1['Credit_History']=df1['Credit_History'].fillna(df1['Credit_History'].mode()[0])


# In[48]:


#changing the data type of each float column to int
from numpy import int64
df['Gender']=df['Gender'].astype('int64')
df['Married']=df['Married'].astype(int64)
df['Dependents']=df['Dependents'].astype(int64)
df['Dependents']=df['Dependents'].astype(int64)
df['Self_Employed']=df['Self_Employed'].astype(int64)
df['CoapplicantIncome']=df['CoapplicantIncome'].astype(int64)
df['LoanAmount']=df['LoanAmount'].astype(int64)
df['Loan_Amount_Term']=df['Loan_Amount_Term'].astype(int64)
df['Credit_History']=df['Credit_History'].astype(int64)


# In[49]:


#changing the data type of each float column to int
from numpy import int64
df1['Gender']=df1['Gender'].astype('int64')
df1['Married']=df1['Married'].astype(int64)
df1['Dependents']=df1['Dependents'].astype(int64)
df1['Dependents']=df1['Dependents'].astype(int64)
df1['Self_Employed']=df1['Self_Employed'].astype(int64)
df1['CoapplicantIncome']=df1['CoapplicantIncome'].astype(int64)
df1['LoanAmount']=df1['LoanAmount'].astype(int64)
df1['Loan_Amount_Term']=df1['Loan_Amount_Term'].astype(int64)
df1['Credit_History']=df1['Credit_History'].astype(int64)


# In[50]:


#Balancing the dataset by using smote
from imbalance.combine import SMOTETomek
smote=SMOTETomek(0.90)


# In[56]:


#dividing the dataset into dependent and independent y and x respectively
from imbalance.combine import SMOTETomek
smote=SMOTETomek(0.90)
y=df['Loan_Status']
x=df.drop(columns=['Loan_Status'],axis=1)
#creating a new x and y variables for the balanced set
x_bal,y_bal=smote.fit_resample(x,y)


# In[58]:


#printing the values of y before balancing the data and after
print(y.value_counts())
print(y_bal.value_counts())


# In[59]:


df.describe()


# In[60]:


df1.describe()


# In[63]:


#Data Visualization using distplot
plt.figure(figsize = (12,5))
plt.subplot(121)
sns.distplot(df['ApplicantIncome'],color='r')
plt.subplot(122)
sns.distplot(df['Credit_History'],color='r')


# In[68]:


#Data Visualization using distplot
plt.figure(figsize = (12,5))
plt.subplot(121)
sns.distplot(df1['ApplicantIncome'],color='r')
plt.subplot(122)
sns.distplot(df1['Credit_History'],color='r')


# In[65]:


#Bivariate analysis
#Data Visualization using countplot
plt.figure(figsize = (18,4))
plt.subplot(1,4,1)
sns.countplot(df['Gender'])
plt.subplot(1,4,2)
sns.countplot(df['Education'])
plt.show


# In[69]:


#Bivariate analysis
#Data Visualization using countplot
plt.figure(figsize = (18,4))
plt.subplot(1,4,1)
sns.countplot(df1['Gender'])
plt.subplot(1,4,2)
sns.countplot(df1['Education'])
plt.show


# In[67]:


#Data Visualization using countplot
plt.figure(figsize = (20,5))
plt.subplot(131)
sns.countplot(df['Married'],hue=df['Gender'])
plt.subplot(132)
sns.countplot(df['Self_Employed'],hue=df['Education'])
plt.subplot(133)
sns.countplot(df['Property_Area'],hue=df['Loan_Amount_Term'])
plt.show


# In[70]:


#Data Visualization using countplot
plt.figure(figsize = (20,5))
plt.subplot(131)
sns.countplot(df1['Married'],hue=df1['Gender'])
plt.subplot(132)
sns.countplot(df1['Self_Employed'],hue=df1['Education'])
plt.subplot(133)
sns.countplot(df1['Property_Area'],hue=df1['Loan_Amount_Term'])
plt.show


# In[72]:


#visualized based gender and income what would be the application status
sns.swarmplot(df['Gender'],df['ApplicantIncome'],hue=df['Loan_Status'])


# In[73]:


#Sclaing the Data 
#performing feature scaling operation using standard scaller an x part of the dataset
#because there different types of values in the columns
sc=StandardScalaer()
x_bal=sc.fit_transform(x_bal)
x_bal=pd.df(x_bal,columns=names)


# In[74]:


#Splitting the Dataset in train and test on balanced dataset
X_train,X_test,y_train,y_test=train_test_split(x_bal,y_bal,test_size=0.33,random_state=42)


# In[75]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[76]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)


# In[78]:


pip install graphviz

pip install pydotplus


# In[79]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())


# In[80]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
RandomForestClassifier()
print(clf.predict([[0, 0, 0, 0]]))


# In[81]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


# In[83]:


dff=pd.read_csv("E:\\NMDS\pl_train.csv")
X = dff.iloc[:, [1, 2, 3]].values
y = dff.iloc[:, -1].values


# In[86]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# In[87]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[88]:


# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[89]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
 


# In[ ]:




