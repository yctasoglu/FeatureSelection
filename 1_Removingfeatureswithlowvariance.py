#FeatureSelector_1

"""
VarianceThreshold is a simple baseline approach to feature selection.
It removes all features whose variance doesn’t meet some threshold.
By default, it removes all zero-variance features,
i.e. features that have the same value in all samples.
"""
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

#Get Sample Data
df = pd.read_excel('Variance_Threshold_Sample.xlsx')
print("Sample Data Frame")
print(df)
def variance_threshold_selector(dataframe, threshold):
    selector = VarianceThreshold(threshold)
    selector.fit(dataframe)
    return dataframe[dataframe.columns[selector.get_support(indices=True)]]
print("Data Frame After Variance Selection")    
print(variance_threshold_selector(dataframe = df, threshold=(.8 * (1 - .8))))
