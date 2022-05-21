#!/usr/bin/env python
# coding: utf-8

# Intro
# 
# 	For this project I have chosen data about heart disease prediction because in the last months I’ve been trying to change my lifestyle and be healthier, and having good habits are probably the most important things we need to improve to accomplishes that change. The data is proposed to help in understanding the relationship between some attributes and having a heart disease.  According to the CDC, heart disease is one of the leading causes of death for people of most races in the US (African Americans, American Indians and Alaska Natives, and white people). About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicator include diabetic status, obesity (high BMI), not getting enough physical activity or drinking too much alcohol. Detecting and preventing the factors that have the greatest impact on heart disease is very important in healthcare. Computational developments, in turn, allow the application of machine learning methods to detect "patterns" from the data that can predict a patient's condition.
#  First at all, my plan is to visualize the data. I would like to see graphs to discover things like noise. My prediction of the data is that if you are a smoker and have bad habits like drink alcohol or not sleeping enough time, you’ll probably suffer a heart disease.
# 
# 
# Description
# 
# The original dataset of nearly 300 variables was reduced to just about 20 variables. In addition to classical EDA, this dataset can be used to apply a range of machine learning methods, most notably classifier models (logistic regression, SVM, random forest, etc.). You should treat the variable "HeartDisease" as a binary ("Yes" - respondent had heart disease; "No" - respondent had no heart disease). But note that classes are not balanced, so the classic model application approach is not advisable. Fixing the weights/under sampling should yield significantly betters results.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("heart_2020_cleaned.csv")
df.head(10)


# In[3]:


df.info()


# In[4]:


heartDisease = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['HeartDisease'] = dataset['HeartDisease'].map(heartDisease)


# In[5]:


age = {"80 or older": 80, "75-79": 75, "70-74": 70, "65-69": 65, "60-64": 60, "55-59": 55, "50-54": 50, "45-49": 45, "40-44": 40, "35-39": 35, "30-34": 30, "25-29": 25, "20-24": 20, "15-19": 15}
data = [df]
for dataset in data:
    dataset['AgeCategory'] = dataset['AgeCategory'].map(age)


# In[6]:


#transfrom the AgeCtegory to Integers
data = [df]
for dataset in data:
    dataset['AgeCategory'] = dataset['AgeCategory'].fillna(0)
    dataset['AgeCategory'] = dataset['AgeCategory'].astype(int)
df.head(10)


# In[7]:


df.isnull().any()


# In[8]:


#smoking convert to 0 = No, 1 = yes
smoker = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['Smoking'] = dataset['Smoking'].map(heartDisease)
df


# In[9]:


#AlcoholDrinking converto to 0 =no, 1 = yes
drinker = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['AlcoholDrinking'] = dataset['AlcoholDrinking'].map(drinker)
df


# In[10]:


#sex converted: Female = 0, Male = 1
gender = {"Female": 0, "Male": 1}
data = [df]
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(gender)
df


# In[11]:


#skinCancer converted: No = 0, Yes = 1
SkinCancer = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['SkinCancer'] = dataset['SkinCancer'].map(SkinCancer)
df


# In[12]:


kidney = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['KidneyDisease'] = dataset['KidneyDisease'].map(kidney)


# In[13]:


asthma = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['Asthma'] = dataset['Asthma'].map(asthma)


# In[14]:


diabetic = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['Diabetic'] = dataset['Diabetic'].map(heartDisease)


# In[15]:


data = [df]
for dataset in data:
    dataset['Diabetic'] = dataset['Diabetic'].fillna(0)
    dataset['Diabetic'] = dataset['Diabetic'].astype(int)


# In[16]:


walkingDiff = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['DiffWalking'] = dataset['DiffWalking'].map(walkingDiff)


# In[17]:


df


# In[18]:


df.describe()


# In[19]:


df.corr()


# In[20]:


df["AgeCategory"].value_counts()


# In[21]:


plt.figure(figsize=(15,7))
sns.countplot(data=df,x="AgeCategory")


# AGE AND SEX
# Create histograms for male (heartDisease vs not heartDisease) and for female (heartDisease vs not heartDisease) where the x axis represents their age and y axis will represent the count of male/female in that age bucket who has heartDisease / not have heartDisease.

# In[22]:


# code goes here for creating the histograms
heartDisease = 'HeartDisease'
not_heartDisease = 'not HeartDisease'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = df[df['Sex']==0]
men = df[df['Sex']==1]
ax = sns.distplot(women[women['HeartDisease']==1].AgeCategory.dropna(), bins=18, label = heartDisease, ax = axes[0], kde =False)
ax = sns.distplot(women[women['HeartDisease']==0].AgeCategory.dropna(), bins=40, label = not_heartDisease, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['HeartDisease']==1].AgeCategory.dropna(), bins=18, label = heartDisease, ax = axes[1], kde = False)
ax = sns.distplot(men[men['HeartDisease']==0].AgeCategory.dropna(), bins=40, label = not_heartDisease, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# In[23]:


FacetGrid = sns.FacetGrid(df, row='GenHealth', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Sex', 'HeartDisease', 'Smoking', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


# In[24]:


# your code for boxplot goes here
sns.barplot(x='Race', y='HeartDisease', data=df)


# In[25]:


# your code for histogram goes here
grid = sns.FacetGrid(df, col='HeartDisease', row='Race', size=2.2, aspect=1.6)
grid.map(plt.hist, 'AgeCategory', alpha=.5, bins=20)
grid.add_legend();


# In[26]:


df


# In[27]:


plt.figure(figsize=(20,7))
sns.barplot(data=df,x="HeartDisease",y="PhysicalHealth",hue="Smoking")


# In[28]:


plt.figure(figsize=(20,7))
sns.barplot(data=df,x="GenHealth",y="PhysicalHealth")


# In[29]:


data = [df]

for dataset in data:
    dataset["SleepTime"] = df["SleepTime"].astype(int)


# In[30]:


stroke = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['Stroke'] = dataset['Stroke'].map(stroke)


# In[31]:


phyA = {"No": 0, "Yes": 1}
data = [df]
for dataset in data:
    dataset['PhysicalActivity'] = dataset['PhysicalActivity'].map(phyA)


# In[32]:


data = [df]

for dataset in data:
    dataset["BMI"] = df["BMI"].astype(int)


# In[33]:


data = [df]

for dataset in data:
    dataset["PhysicalHealth"] = df["PhysicalHealth"].astype(int)


# In[34]:


data = [df]

for dataset in data:
    dataset["MentalHealth"] = df["MentalHealth"].astype(int)


# In[35]:


genH = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
data = [df]
for dataset in data:
    dataset['GenHealth'] = dataset['GenHealth'].map(genH)


# In[36]:


data = [df]
for dataset in data:
    dataset.loc[ dataset['AgeCategory'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['AgeCategory'] > 11) & (dataset['AgeCategory'] <= 18), 'AgeCategory'] = 1
    dataset.loc[(dataset['AgeCategory'] > 18) & (dataset['AgeCategory'] <= 22), 'AgeCategory'] = 2
    dataset.loc[(dataset['AgeCategory'] > 22) & (dataset['AgeCategory'] <= 30), 'AgeCategory'] = 3
    dataset.loc[(dataset['AgeCategory'] > 30) & (dataset['AgeCategory'] <= 40), 'AgeCategory'] = 4
    dataset.loc[(dataset['AgeCategory'] > 40) & (dataset['AgeCategory'] <= 50), 'AgeCategory'] = 5
    dataset.loc[(dataset['AgeCategory'] > 50) & (dataset['AgeCategory'] <= 70), 'AgeCategory'] = 6
    dataset.loc[ dataset['AgeCategory'] > 70, 'AgeCategory'] = 7


# In[37]:


data = [df]
for dataset in data:
     dataset.loc[ dataset['AgeCategory'] > 70, 'AgeCategory'] = 7


# In[38]:


df


# BUILDING MACHINE LEARNING MODEL

# In[39]:


# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# In[40]:


df = df.drop("Race", axis = 1)


# In[41]:


df = df.drop("MentalHealth", axis = 1)


# In[42]:


df = df.drop("PhysicalHealth", axis = 1)


# In[43]:


df = df.drop("Age", axis = 1)


# In[44]:


df


# In[45]:


from sklearn.model_selection import train_test_split
x = df.drop("HeartDisease",axis=1)
y = df["HeartDisease"]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# In[46]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[47]:


# Random Forest:
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


# In[48]:


# Logistic Regression:
# your code goes here
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)


# In[49]:


# K Nearest Neighbor:
# your code goes here
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, y_train)  
Y_pred = knn.predict(X_test)  
acc_knn = round(knn.score(X_train, y_train) * 100, 2)


# In[50]:


# Gaussian Naive Bayes:
# your code goes here
gaussian = GaussianNB() 
gaussian.fit(X_train, y_train)  
Y_pred = gaussian.predict(X_test)  
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)


# In[51]:


# Perceptron:
# your code goes here
perceptron = Perceptron(max_iter=25)
perceptron.fit(X_train, y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)


# In[52]:


# Linear Support Vector Machine:
# your code goes here
linear_svc = LinearSVC(max_iter = 100000)
linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)


# In[53]:


# Decision Tree
# your code goes here
decision_tree = DecisionTreeClassifier() 
decision_tree.fit(X_train, y_train)  
Y_pred = decision_tree.predict(X_test)  
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)


# Which model is the best?

# In[54]:


# your code goes here for comparing the accuracy of the above 7 models
print("Random Forest: ", acc_random_forest)
print("Logistic Regression: ", acc_log)
print("K Nearest Neighbor: ", acc_knn)
print("Gaussian Naive Bayes: ", acc_gaussian)
print("Perceptron: ", acc_perceptron)
print("Linear Support Vector Machine: ", acc_linear_svc)
print("Decision tree: ", acc_decision_tree)


# Random forest with K-fold cross validation

# In[56]:


from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# This looks much more realistic than before. Our model has a average accuracy of 90% with a standard deviation of 0.15 %. The standard deviation shows us, how precise the estimates are .
# 
# This means in our case that the accuracy of our model can differ + — 0.15%. rally low

# In[57]:


# Your code goes here, you can create more cells if you want show the results using a dataframe
lr = LogisticRegression()
scores_lr = cross_val_score(lr, X_train, y_train, cv=10, scoring = "accuracy")

kn = KNeighborsClassifier(n_neighbors = 3) 
scores_kn = cross_val_score(kn, X_train, y_train, cv=10, scoring = "accuracy")

ga = GaussianNB()
scores_ga = cross_val_score(ga, X_train, y_train, cv=10, scoring = "accuracy")

per = Perceptron(max_iter=25)
scores_per = cross_val_score(per, X_train, y_train, cv=10, scoring = "accuracy")

lin = LinearSVC(max_iter = 100000)
scores_lin = cross_val_score(lin, X_train, y_train, cv=10, scoring = "accuracy")

dt = DecisionTreeClassifier() 
scores_dt = cross_val_score(dt, X_train, y_train, cv=10, scoring = "accuracy")


# In[58]:


results = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 
              'Naive Bayes', 'Perceptron', 
              'Linear SVC', 
              'Decision Tree'],
    'Scores': [scores_kn, scores_lr, 
              scores_ga, scores_per, scores_lin, 
              scores_dt],
    'Mean': [scores_kn.mean(), scores_lr.mean(), scores_ga.mean(), scores_per.mean(), scores_lin.mean(), scores_dt.mean()],
    'Std': [scores_kn.std(), scores_lr.std(), scores_ga.std(), scores_per.std(), scores_lin.std(), scores_dt.std()]})

results


# Importance features:

# In[68]:



from matplotlib import style
importances = pd.DataFrame({'feature': x.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head()


# In[69]:


importances.plot.bar()


# Find the features that does not play a significant role in the random forest classifiers prediction process. Drop those features and model the Random Forest classifier again to see if it improves or not and report your findings.

# In[70]:


# your code goes here
df  = df.drop("AlcoholDrinking", axis=1)
df  = df.drop("SkinCancer", axis=1)
df  = df.drop("KidneyDisease", axis=1)
df  = df.drop("Asthma", axis=1)


# In[71]:


random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")


# the result after dropping the less important features does not improve, remains the same.

# In[ ]:




