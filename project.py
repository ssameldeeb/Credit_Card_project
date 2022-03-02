import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

data = pd.read_csv("Credit Card Defaulter Prediction.csv")

print(data.shape)
print(data.columns.values)
print(data.isnull().sum())
print(data.head())
print(data.dtypes)

print(data["SEX"].value_counts())
print("_"*100)
print(data["EDUCATION"].value_counts())
print("_"*100)
print(data["MARRIAGE"].value_counts())

sns.countplot(data["Default"])
plt.show()

La = LabelEncoder()

data["SEX"] = La.fit_transform(data["SEX"])
data["EDUCATION"] = La.fit_transform(data["EDUCATION"])
data["MARRIAGE"] = La.fit_transform(data["MARRIAGE"])

data["Default"] = data["Default"].map({"N":0,"Y":1})

print(data.dtypes)

plt.figure(figsize=(14,6))
sns.heatmap(data.corr(),annot=True)
plt.show()

x = data.drop("Default",axis=1)
y = data["Default"]

ss = StandardScaler()
x = ss.fit_transform(x)
print(x[:5])


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=44, shuffle =True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

Lo = LogisticRegression()
Lo.fit(X_train, y_train)

print("_"*100)
print(Lo.score(X_train, y_train))
print(Lo.score(X_test, y_test))
print("_"*100)


# print("_"*150)
# for x in range(2,20):
#     Dt = DecisionTreeClassifier(max_depth=x,random_state=33)
#     Dt.fit(X_train, y_train)

#     print("x = ", x)
#     print(Dt.score(X_train, y_train))
#     print(Dt.score(X_test, y_test))
#     print("_"*100)
    


# print("_"*150)
# MLC = MLPClassifier(activation='tanh',
#                                  solver='lbfgs', 
#                                  learning_rate='constant',
#                                  alpha=0.00001 ,hidden_layer_sizes=(200, 3),random_state=33)


# MLC.fit(X_train, y_train)

# print("_"*100)
# print(MLC.score(X_train, y_train))
# print(MLC.score(X_test, y_test))

# print("_"*150)
# for x in range(2,30):
#     KNN = KNeighborsClassifier(n_neighbors=x)

#     print(": " ,x)
#     KNN.fit(X_train, y_train)
#     print(KNN.score(X_train, y_train))
#     print(KNN.score(X_test, y_test))

Dt = DecisionTreeClassifier(max_depth=3,random_state=33)
Dt.fit(X_train, y_train)

print("_"*100)
print(Dt.score(X_train, y_train))
print(Dt.score(X_test, y_test))
print("_"*100)

y_pred = Dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm,annot=True)
plt.show()


autput = pd.DataFrame({"y_test":y_test, "y_pred":y_pred})
# autput.to_csv("autput.csv",index=False)