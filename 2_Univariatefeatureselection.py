#Univariatefeatureselection

"""
Univariate feature selection works by selecting the best features
based on univariate statistical tests. It can be seen as a preprocessing
step to an estimator. Scikit-learn exposes feature selection routines
as objects that implement the transform method:
"""
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

#Get Sample Data
df = pd.read_excel('Chi.xlsx')
print("Sample Data Frame")
print(df)
y = df['f4']
X = df.drop(['f4'],axis=1)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)

# Create and fit selector
selector = SelectKBest(chi2, k=2)
selector.fit(X, y)
# Get columns to keep and create new dataframe with those only
cols = selector.get_support(indices=True)
features_df_new = df.iloc[:,cols]
print(features_df_new)