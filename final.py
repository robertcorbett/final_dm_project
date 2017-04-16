# Script for final project
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.cluster import KMeans

data = pd.read_csv('./datasets/vgsales_with_ratings.csv')
data = data.dropna()

# print data.head()
# print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
# data = data.drop(['Platform', 'Year_of_Release', 'Genre', 'Publisher', 'Developer', 'Rating', ], axis=1)
data = data[['Critic_Score', 'Global_Sales']]
print data.head()

kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
result = kmeans.fit(data)
data["label"] = result.labels_.astype(str)
print result.labels_ 
# sns.pairplot(data, hue="category", diag_kind="kde")

import seaborn as sns; sns.set()
# datset is loaded as a pandas dataframe
# flights = sns.load_dataset("flights")
vg = sns.load_dataset(data)
vg = vg.pivot("month", "year", "passengers")
g = sns.clustermap(vg)