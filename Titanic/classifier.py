import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
#загружаем данные
df_train = pd.read_csv("titanic/train.csv")
df_test = pd.read_csv("titanic/test.csv")

#Выводим сводку о входных данных
#print(df_train.describe())

#Выводим данным в которых есть пустые значения
null_data = (df_train.isnull().sum()/df_train.isnull().count())*100
print(null_data.sort_values(ascending=False).head(3))
#Выводим данным в которых есть пустые значения для Test
null_data = (df_test.isnull().sum()/df_test.isnull().count())*100
print(null_data.sort_values(ascending=False).head(3))
#Строим HISTограмы по каждому полю и построение графика
#df_train.hist(figsize=(15,20))
#plt.show()

#Filling Missing data and transforming features
X = df_train.iloc[:,1:]

df_train["Embarked"].fillna(df_train["Embarked"][0], inplace=True)

df_test["Fare"].fillna(df_test["Fare"].mean(), inplace=True)

def age_transform(df):
    df["Age"].fillna(-0.5, inplace=True)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df["Age"], bins, labels=group_names)
    df["Age"] = categories
    return df

def name_transform(df):
    df["Lname"] = df["Name"].apply(lambda x: x.split(',')[0])
    df["prefix"] = df["Name"].apply(lambda x: x.split(' ')[1])
    return df

def feature_transform(df):
    df = name_transform(df)
    df = age_transform(df)
    df.drop(["PassengerId","Name", "Ticket", "Cabin"], axis=1, inplace=True)
    return df

df_train = feature_transform(df_train)

df_test = feature_transform(df_test)

X = df_train.iloc[:, 1:]
y = df_train.iloc[:, 0]
X_test =  df_test

def features_encode(df):
    features = df.columns
    for feature in features:
        le = LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df

X= features_encode(X)
X_test =  features_encode(X_test)

#Преобразование данных
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.fit_transform(X_test)


X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size = 0.2, random_state = 0)
clf = XGBClassifier(n_estimators=210, max_depth=6,gamma=0.05)
clf.fit(X_train, y_train)

xgb_pred = clf.predict(X_cv)
accuracy = accuracy_score(y_cv, xgb_pred)
clf_report = classification_report(y_cv, xgb_pred)
print("XGB Classifier Accuracy: ", accuracy*100, "%", "\n\n",  "XGB Classification Report: \n ", clf_report)

y_pred = clf.predict(X_test)
df = pd.read_csv("titanic/gender_submission.csv")
df["Survived"] = y_pred
df.set_index(['PassengerId', 'Survived'], drop=True, inplace=True, )
df.to_csv("titanic/titanic_disaster.csv")