# Script for final project
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.cluster import KMeans

data = pd.read_csv('./datasets/vgsales_with_ratings.csv')
happy = pd.read_csv('./datasets/happy2016.csv')
data = data.dropna()

# print data.head()
# print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
# data = data.drop(['Platform', 'Year_of_Release', 'Genre', 'Publisher', 'Developer', 'Rating', ], axis=1)
sns.set_style("ticks")
data = data[['Critic_Score', 'Global_Sales']]

print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
print happy.head()
print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
print data.head()
print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
# ax = sns.pointplot(x="Global_Sales", y="Critic_Score", data=data)
# ax = sns.jointplot(x="Global_Sales", y="Critic_Score", data=data)
ax = sns.jointplot(x="Critic_Score", y="Global_Sales", data=data)
sns.plt.show()

kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
games_result = kmeans.fit(data)
# happy_result = kmeans.fit(happy)
data["label"] = result.labels_.astype(str)
print games_result.labels_ 
# print happy_result.labels_ 
# sns.pairplot(data, hue="category", diag_kind="kde")

# datset is loaded as a pandas dataframe
# flights = sns.load_dataset("flights")

# sns.set()
# vg = sns.load_dataset(data)
# vg = vg.pivot("month", "year", "passengers")
# g = sns.clustermap(vg)
