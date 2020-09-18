'''<-----------Topic: Car Sales Purchase Prediction---------------->'''

#▪ Importing necessary libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#loading data
data = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')


print('Some Dataset Samples are:\n',data.head())

data = data.rename(columns = {'Car Purchase Amount':'target'})

print('Some Dataset samples after renaming the target column:\n',data.head())



#▪ Visualize Data

import matplotlib.pyplot as plt
import seaborn as sns

sns.pairplot(data)
plt.title('Pairplot between the features' )
plt.show()
sns.barplot(data.iloc[:,3],data['target'])
plt.title('Barplot between target and Gender')
plt.show()
for i in range(4,8):
    sns.jointplot(data['target'],data.iloc[:,i],kind='reg')
    plt.show()


fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(221)
ax2= fig.add_subplot(222)
ax3= fig.add_subplot(223)
ax4= fig.add_subplot(224)


ax1.boxplot('Age',data = data)
ax1.set_title('Age boxplot')
ax2.boxplot('Annual Salary',data = data)
ax2.set_title('Annual Salary boxplot')
ax3.boxplot('Credit Card Debt',data = data)
ax3.set_title('Credit Card Debt boxplot')
ax4.boxplot('Net Worth',data = data)
ax4.set_title('Net Worth boxplot')

plt.show()

#Data cleaning

print(data.describe())
print(data.info())

print('Checking for null value in the dataset\n')
print(data.isnull().all())


#Correlation heatmap
corr= data.corr()

sns.heatmap(corr,annot=True)
plt.title('Correlation heatmap')
plt.show()

data = data.drop(columns=['Customer Name', 'Customer e-mail', 'Country','Gender','Credit Card Debt'])
print('Dataset columns after some columns drops out:\n',data.columns)


#Normalize the dataset

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
stddata = sc.fit_transform(data)

X = stddata[:,0:3]
y = data['target']



#Splits the dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)


#Training the Model


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
model = lm.fit(X_train,y_train)
score = model.score(X_test,y_test)
pred = model.predict(X_test)

print('Compairing')
print('First Y test samples are:',y_test.head())
print('Some predicted values are:',pred[0:5])

sns.lineplot(x = pred, y = y_test ,marker  = 'o',markersize=4)
plt.xlabel('Predicted values')
plt.ylabel('target values')
plt.title('Comparision between predicted value and original value')
plt.show()

print('Accuracy score of the model is:',score)
