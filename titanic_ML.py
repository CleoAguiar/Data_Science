import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.tree import DecisionTreeClassifier

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))

# Input data files
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

# Seeing 5 rows in train file
train.head()

# Dop Name, Ticket, Cabin collumn
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Creating new_data
new_data_train = pd.get_dummies(train)
new_data_test = pd.get_dummies(test)

train.head()
new_data_train.head()

# Seeing null values
new_data_train.isnull().sum().sort_values(ascending=False).head(10)

# Replace null values
new_data_train['Age'].fillna(new_data_train['Age'].mean(), inplace=True)
new_data_test['Age'].fillna(new_data_test['Age'].mean(), inplace=True)

new_data_test.isnull().sum().sort_values(ascending=False).head(10)

# Replace null value
new_data_test['Fare'].fillna(new_data_test['Fare'].mean(), inplace=True)

# X is input and Y is output
x = new_data_train.drop('Survived', axis=1)
y = new_data_train['Survived']

# Model Tree Classifier
tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x, y)

# Seeing score about training seads
tree.score(x,y)

submission = pd.DataFrame()
submission['PassengerId'] = new_data_test['PassengerId']
submission['Survived'] = tree.predict(new_data_test)

submission.to_csv('submission.csv', index=False)
