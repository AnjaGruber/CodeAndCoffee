#!/usr/bin/env python
# coding: utf-8

# ### Collecting data, importing libs

# In[78]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
sns.set(style="ticks")

df = pd.read_excel(r'C:\Users\Anja\Desktop\Data mining Anja 36119\CoffeeAndCodeLT2018.xlsx')
df.head()


# In[79]:


print("number of data: "+ str(len(df.index)))


# In[80]:


df.dtypes


# In[81]:


df.describe(include="all")


# ### Analyzing Data

# In[82]:


sns.countplot(x = "CoffeeCupsPerDay", data = df)


# In[83]:


CoffeeCupsPerDay = df.CoffeeType
CoffeeCupsPerDay.value_counts().plot(kind='barh')
print (CoffeeCupsPerDay.value_counts(normalize=True))


# In[84]:


CoffeeCupsPerDay = df["CoffeeCupsPerDay"]
print (CoffeeCupsPerDay.value_counts())
prob= CoffeeCupsPerDay.value_counts(normalize=True)
threshold = 0.02
mask = prob > threshold
tail_prob = prob.loc[~mask].sum()
prob = prob.loc[mask]
prob['other'] = tail_prob
prob.plot(kind='bar')
plt.xticks(rotation=25)
plt.show()


# In[85]:


g = sns.catplot(x="CoffeeCupsPerDay", hue="CoffeeType",
                data=df, kind="count",
                height=4, aspect=.99, sharex=False, sharey=False, palette="RdBu_r");


# In[86]:


#Da li žene ili muškarci količinski više konzumiraju kafu?

cups_by_gender = df.pivot_table(columns='Gender', values='CoffeeCupsPerDay', aggfunc={'mean', 'count', 'min', 'max'})
cups_by_gender.head()


# In[87]:


sns.countplot(x = "CoffeeCupsPerDay", hue = "Gender", data = df)


# In[88]:


# Problem: Da li je uticaj kafe isti na muškarce i žene?

df.groupby(['Gender', 'CodingWithoutCoffee', 'CoffeeSolveBugs']).CoffeeCupsPerDay.agg(['mean','count', 'min', 'max'])


# In[89]:


# Problem: Da li je uticaj kafe isti na muškarce i žene?

group_by_gender = df.groupby(['Gender', 'CodingWithoutCoffee', 'CoffeeSolveBugs'])
group_by_gender.size().unstack().reset_index()


# In[90]:


# Outliners - greske prilikom davanja podataka, osobe koje kazu da ne piju kafu dok kodiraju za vreme konzumiranja stave dok kodiraju

df.groupby(['Gender', 'CodingWithoutCoffee', 'CoffeeSolveBugs', 'CoffeeTime']).CoffeeCupsPerDay.agg(['mean','count', 'min', 'max'])


# In[91]:


g = sns.catplot(x="Gender", hue="CoffeeSolveBugs", col="CodingWithoutCoffee",
                data=df, kind="count",
                height=4, aspect=.7, sharex=False, sharey=False, palette="Pastel1");


# In[92]:


sns.catplot(x="AgeRange",y="CoffeeCupsPerDay",data=df,hue="Gender",aspect=2.5,kind="point")


# In[93]:


#Da li različite vrste kafe daju različiti uticaj na uspešnost programiranja?

plt.figure(figsize=(8,4))
ax = sns.countplot(x = "CoffeeType", hue = "CoffeeSolveBugs", data = df, palette="Pastel1") 
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# In[94]:


plt.figure(figsize=(8,4))
ax = sns.countplot(x = "CoffeeType", hue = "CodingHours", data = df, palette="colorblind") 
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.gca().legend(loc='center left', bbox_to_anchor=(1, 0.5), title='CodingHours')
plt.show()


# In[95]:


sns.FacetGrid(hue="Gender",data=df,aspect=2.5,height=5).map(sns.kdeplot,"CodingHours",shade=True).add_legend()


# In[96]:


df.boxplot('CoffeeCupsPerDay','Gender',rot = 30,figsize=(5,6))


# In[150]:


df.boxplot('CoffeeCupsPerDay','CodingHours',rot = 30,figsize=(5,6))


# In[97]:


df.boxplot('CoffeeCupsPerDay','CoffeeTime',rot = 30,figsize=(5,6))


# In[98]:


df.boxplot('CoffeeCupsPerDay','CoffeeType',rot = 30,figsize=(5,6))


# In[99]:


df.boxplot('CoffeeCupsPerDay','AgeRange',rot = 30,figsize=(5,6))


# In[100]:


df.boxplot('CoffeeCupsPerDay','CoffeeSolveBugs',rot = 30,figsize=(5,6))


# In[101]:


sns.lmplot(x="CodingHours",y="CoffeeCupsPerDay",hue="Gender",data=df)


# In[102]:


sns.lmplot(x="CodingHours",y="CoffeeCupsPerDay",hue="CoffeeSolveBugs",data=df)


# ### Data Wrangling

# In[103]:


print(df.isnull().sum())


# In[104]:


sns.heatmap(df.isnull(), yticklabels=False, cbar=False)


# In[105]:


df['CoffeeType'] = df['CoffeeType'].fillna(df['CoffeeType'].value_counts().index[0])


# In[106]:


print(df.isnull().values.sum())


# In[107]:


df['AgeRange'] = df['AgeRange'].fillna(df['AgeRange'].value_counts().index[0])


# In[108]:


print(df.isnull().values.sum())


# In[109]:


print("number of data: "+ str(len(df.index)))


# In[110]:


df.drop('Country', axis=1, inplace=True)


# In[111]:


df.head()


# In[112]:


df.dtypes


# In[113]:


df["CoffeeTime"] = df["CoffeeTime"].astype('category')
df["CodingWithoutCoffee"] = df["CodingWithoutCoffee"].astype('category')
df["CoffeeSolveBugs"] = df["CoffeeSolveBugs"].astype('category')
df["CoffeeType"] = df["CoffeeType"].astype('category')
df.dtypes


# In[114]:


lb_make = LabelEncoder()
df["CoffeeType_code"] = lb_make.fit_transform(df["CoffeeType"])


# In[115]:


code = df["CoffeeType_code"].unique()
real =  df["CoffeeType"].unique()
print('CoffeeType')
print(' ')
for c,r in zip(code,real):
    print( '{:>0}     {:<15}'.format(c,r))


# In[116]:


df["CodingWithoutCoffee_code"] = df["CodingWithoutCoffee"].cat.codes


# In[117]:


code = df["CodingWithoutCoffee_code"].unique()
real =  df["CodingWithoutCoffee"].unique()
print('CodingWithoutCoffee')
print(' ')
for c,r in zip(code,real):
    print( '{:>0}     {:<15}'.format(c,r))


# In[118]:


df["CoffeeSolveBugs_code"] = df["CoffeeSolveBugs"].cat.codes


# In[119]:


code = df["CoffeeSolveBugs_code"].unique()
real =  df["CoffeeSolveBugs"].unique()
print('CoffeeSolveBugs')
print(' ')
for c,r in zip(code,real):
    print( '{:>0}     {:<15}'.format(c,r))


# In[120]:


df.head()


# In[121]:


df["Gender_code"] = np.where(df["Gender"].str.contains("Fe"), 0, 1)


# In[122]:


code = df["Gender_code"].unique()
real =  df["Gender"].unique()
print('Gender')
print(' ')
for c,r in zip(code,real):
    print( '{:>0}     {:<15}'.format(c,r))


# In[123]:


df_age = pd.DataFrame({'AgeRange': ['12 to 17', '18 to 29', '30 to 39', '40 to 49','50 to 59']})

def split_mean(x):
    split_list = x.split(' to ')
    mean = (float(split_list[0])+float(split_list[1]))/2
    return mean

df['Age_mean'] = df['AgeRange'].apply(lambda x: split_mean(x))


# In[124]:


code = df["Age_mean"].unique()
real =  df["AgeRange"].unique()
print('AgeRange')
print(' ')
for c,r in zip(code,real):
    print( '{:>0}     {:<15}'.format(c,r))


# In[125]:


df['CoffeeTime_code'] = df['CoffeeTime'].map( {'No specific time':0, 'In the morning':1, 'Before coding':2, 
                                               'Before and while coding':3, 'While coding':4, 'After coding':5, 
                                               'All the time':6  })


# In[126]:


code = df["CoffeeTime_code"].unique()
real =  df["CoffeeTime"].unique()
print('CoffeeTime')
print(' ')
for c,r in zip(code,real):
    print( '{:>0}     {:<15}'.format(c,r))


# In[127]:


df.drop('AgeRange', axis=1, inplace=True)
df.drop('CodingWithoutCoffee', axis=1, inplace=True)
df.drop('CoffeeSolveBugs', axis=1, inplace=True)
df.drop('CoffeeType', axis=1, inplace=True)
df.drop('Gender', axis=1, inplace=True)
df.drop('CoffeeTime', axis=1, inplace=True)


# In[128]:


df.head()


# ### Model Development, Train data, Predictions

# In[129]:


df.corr()


# In[130]:


prediction_df = df[['CoffeeCupsPerDay','Gender_code', 'CodingHours']]
prediction_df.head()


# In[131]:


prediction_df.corr()


# In[132]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = prediction_df.values
X, y = data[:,1:3], data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred =classifier.predict(X_test)

accuracy_score(y_test, y_pred)*100


# In[144]:


#Accuracy of Model with Cross Validation 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

data = prediction_df.values
X, y = data[:,1:3], data[:, 0]

model = svm.SVC(C=1, kernel='linear')
accuracy = cross_val_score(model, X, y, scoring='accuracy', cv = 2)
print(accuracy)
#get the mean of each fold 
print("Accuracy of Model with Cross Validation is:",accuracy.mean() * 100)

model.fit(X, y)
print('Broj kafa: ', model.predict([[1, 6]]))


# In[134]:


from sklearn.model_selection import cross_val_predict
from sklearn import linear_model

lr = linear_model.LinearRegression()
# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validation:
predicted = cross_val_predict(lr, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, predicted, edgecolors=(0, 0, 0))
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()


# In[135]:


from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

data = prediction_df.values
X, y = data[:,1:3], data[:, 0]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 7)


models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))



results = []
names = []
for name, model in models:
    accuracy = cross_val_score(model, X, y, scoring='accuracy', cv = 2)
    print(accuracy)
    #get the mean of each fold 
    print("Accuracy of", name, "Model with Cross Validation is:",accuracy.mean() * 100)
    
   


# ### Application

# In[159]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

data = prediction_df.values
X, y = data[:,1:3], data[:, 0]
model = svm.SVC(C=1, kernel='linear')
accuracy = cross_val_score(model, X, y, scoring='accuracy', cv = 2)

model.fit(X, y)


root= tk.Tk()
root.title("Predvidjanje")
root.iconbitmap(r'C:\Users\Anja\Desktop\Data mining Anja 36119\favicon.ico')

canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()

label1 = tk.Label(root, text='Type CodingHours: ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

label2 = tk.Label(root, text=' Type Gender_code (0 - female, 1 - male): ')
canvas1.create_window(100, 120, window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

def values(): 
    global New_CodingHours #our 1st input variable
    New_CodingHours = float(entry1.get()) 
    
    global New_Gender_code #our 2nd input variable
    New_Gender_code = float(entry2.get()) 
    
    Prediction_result  = ('Predicted CoffeeCupsPerDay: ', model.predict([[New_CodingHours ,New_Gender_code]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='#c7f2fc')
    canvas1.create_window(260, 190, window=label_Prediction)

button1 = tk.Button (root, text='Predict CoffeeCupsPerDay',command=values, bg='#c7f2fc') 
canvas1.create_window(270, 150, window=button1)

#plot 1st scatter 
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(prediction_df['CodingHours'].astype(float),df['CoffeeCupsPerDay'].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend(['CoffeeCupsPerDay']) 
ax3.set_xlabel('CodingHours')
ax3.set_title('CodingHours Vs. CoffeeCupsPerDay')

#plot 2nd scatter 
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['Gender_code'].astype(float),df['CoffeeCupsPerDay'].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend(['CoffeeCupsPerDay']) 
ax4.set_xlabel('Gender_code')
ax4.set_title('Gender_code Vs. CoffeeCupsPerDay')

root.mainloop()

