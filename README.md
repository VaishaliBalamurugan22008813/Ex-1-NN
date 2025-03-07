<H3>ENTER YOUR NAME:VAISHALI BALAMURUGAN</H3>
<H3>ENTER YOUR REGISTER NO.:212222230164</H3>
<H3>EX. NO.1</H3>
<H3>DATE:7/3/2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#importing required libraries
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np

#loading dataset
df=pd.read_csv("archive.zip")
print(df)

#Splitting Features and Target Variables
x=df.iloc[:,:-1].values
print(x)

#Handling Missing Values
y=df.iloc[:,-1].values
print(y)
print(df.isnull().sum())

numeric_df = df.select_dtypes(include=np.number)
df[numeric_df.columns]=df[numeric_df.columns].fillna(numeric_df.mean().round(1))
print(df.isnull().sum())

y = df.iloc[:, -1].values
print(y)
df.duplicated()

if 'Calories' in df.columns:
  print(df['Calories'].describe())
else:
  print("Column 'Calories' not found in the DataFrame.")

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))
```
## OUTPUT:
![image](https://github.com/user-attachments/assets/459650cb-4ae1-4be3-87c7-1ae11ec1d441)
![image](https://github.com/user-attachments/assets/e141717e-c958-4e2d-8b36-46e45b6698d3)
![image](https://github.com/user-attachments/assets/ea75b4b5-9ee0-49e2-980e-0a68e16a3610)
![image](https://github.com/user-attachments/assets/750773bb-9a3d-463a-b9cd-d9d972e23b28)
![image](https://github.com/user-attachments/assets/39e05fc4-4692-436a-b407-e837cf9f51dd)
![image](https://github.com/user-attachments/assets/654ba1f4-630e-407c-8540-75aa3e4ca560)
![image](https://github.com/user-attachments/assets/bc927c78-a598-4d9e-9a08-a5315a198ddd)
![image](https://github.com/user-attachments/assets/c2c38496-18f3-442e-98ec-aba612551129)







## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


