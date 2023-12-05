# %% Importing the libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
import seaborn as sn
import xgboost as xg
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# %% Loading the dataset:

original = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/bankloan.csv")
data = pd.read_csv("C:/Users/umesh/OneDrive/Desktop/Umesh/Data Analysis/bankloan.csv")

# %% Getting the basic infos:

stats = data.describe()
print(data.info())
print(data.isna().sum())
print(data.shape)

# %% Detecting outliers: (by Boxplot)

sn.boxplot(data=data)
plt.xticks(rotation="vertical")
plt.show()

# %% Removing these outliers:

outliers = boxplot_stats(data[data.columns[[3, 4, 6, 8]]])
outliers = pd.DataFrame(outliers)['fliers']
data = data[~data['Income'].isin(outliers[0])]
data = data[~data['ZIP.Code'].isin(outliers[1])]
data = data[~data['CCAvg'].isin(outliers[2])]
data = data[~data['Mortgage'].isin(outliers[3])]

# %% MinMax Scalling

minmax = MinMaxScaler()
maxing = ['Age', 'Experience', 'Income', 'Mortgage']
for i in maxing:
    data[i] = minmax.fit_transform(data[i].to_numpy().reshape((-1,1)))

# %% Splitting into Dependent and Independent:(Prediction of Credit Card)

X = data.iloc[:, 1:13]
Y = data.iloc[:, 13]

# %% Splitting the data into Train and Test:

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

# %% Fitting into Machine Learning Models:

# Logistic Regression:

lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train.ravel())
Y_predict = lr.predict(X_test)
lr1 = lr.score(X_train, y_train)

# %% Gradient Boosting Classifier:

gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, random_state=100, max_features=5)
gbc.fit(X_train, y_train.ravel())
Y_predict = gbc.predict(X_test)
gbc1 = gbc.score(X_train, y_train)

# %% Random Forest Classifier:

Rf = RandomForestClassifier(n_jobs=-1, n_estimators=500, random_state=100, max_features=5)
Rf.fit(X_train, y_train.ravel())
Y_predict = Rf.predict(X_test)
rf1 = Rf.score(X_train, y_train)

# %% XGB Classifer:

xgb_r = xg.XGBRegressor(n_estimators=1000, verbosity=3, n_jobs=-1)
xgb_r.fit(X_train, y_train)
Y_predict = xgb_r.predict(X_test)
xbr1 = xgb_r.score(X_train, y_train)

# %% Tabulation of the results:

result = pd.DataFrame([lr1, gbc1, rf1, xbr1],["Logistic Regression", "Gradient Boosting Classifier", "Random Forest Classifier", "XGB Classifer"], columns=['Scores'])
result = result.sort_values(by="Scores", ascending=False)

# %% Graphical Representation:

plt.figure(figsize=(10, 7))
sn.barplot(data=result['Scores'], alpha=0.5, edgecolor='black', color='lightgreen')
plt.xticks(rotation=0)
plt.show()

# %%