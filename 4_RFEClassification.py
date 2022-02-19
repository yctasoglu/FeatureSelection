#4_RFEClassification.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train)

df_train.drop(['Ticket', 'PassengerId'], axis=1, inplace=True)

gender_mapper = {'male': 0, 'female': 1}
df_train['Sex'].replace(gender_mapper, inplace=True)

df_train['Title'] = df_train['Name'].apply(lambda x: x.split(',')[1].strip().split(' ')[0])
df_train['Title'] = [0 if x in ['Mr.', 'Miss.', 'Mrs.'] else 1 for x in df_train['Title']]
df_train = df_train.rename(columns={'Title': 'Title_Unusual'})
df_train.drop('Name', axis=1, inplace=True)

df_train['Cabin_Known'] = [0 if str(x) == 'nan' else 1 for x in df_train['Cabin']]
df_train.drop('Cabin', axis=1, inplace=True)

emb_dummies = pd.get_dummies(df_train['Embarked'], drop_first=True, prefix='Embarked')
df_train = pd.concat([df_train, emb_dummies], axis=1)
df_train.drop('Embarked', axis=1, inplace=True)

df_train['Age'] = df_train['Age'].fillna(int(df_train['Age'].mean()))



correlated_features = set()
correlation_matrix = df_train.drop('Survived', axis=1).corr()

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            correlated_features.add(colname)
print(correlated_features)

X = df_train.drop('Survived', axis=1)
target = df_train['Survived']

rfc = RandomForestClassifier(random_state=101)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, target)

print('Optimal number of features: {}'.format(rfecv.n_features_))

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.show()

print(np.where(rfecv.support_ == False)[0])

X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

print(rfecv.estimator_.feature_importances_)

dset = pd.DataFrame()
dset['attr'] = X.columns
dset['importance'] = rfecv.estimator_.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)


plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()