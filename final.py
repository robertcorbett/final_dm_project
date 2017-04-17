# Script for final project
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn import cluster

data = pd.read_csv('./datasets/vgsales_with_ratings.csv')
happy = pd.read_csv('./datasets/happy2016.csv')
data = data.dropna()
print (len(data))

# print data.head()
# print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
# data = data.drop(['Platform', 'Year_of_Release', 'Genre', 'Publisher', 'Developer', 'Rating', ], axis=1)
data = data[['Critic_Score', 'Global_Sales']]
happy_data = happy.drop(['Country', 'Region', 'Happiness Rank', 'Upper Confidence Interval', 'Lower Confidence Interval'], axis=1)

print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print (happy_data.head())
print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print (data.head())
print ('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# ax = sns.pointplot(x="Global_Sales", y="Critic_Score", data=data)
# ax = sns.jointplot(x="Global_Sales", y="Critic_Score", data=data)
# ax = sns.jointplot(x="Critic_Score", y="Global_Sales", data=data)
# sns.plt.show()

# kmeans = cluster.KMeans(init='k-means++', n_clusters=4, n_init=10)
# spec = cluster.SpectralClustering(n_clusters=4, eigen_solver='arpack', affinity='nearest_neighbors')
affinity_prop = cluster.AffinityPropagation(damping=.9, preference=-200)
# happy_result = kmeans.fit(happy_data)
# happy_result = spec.fit(happy_data)
happy_result = affinity_prop.fit(happy_data)


# data["label"] = games_result.labels_.astype(str)
# print (len(games_result.labels_))
# print (games_result)

# print (len(happy_result.labels_))
# print happy_result.labels_

happy['label'] = happy_result.labels_
# print happy.head()
sns.set_style("ticks")

# sns.pairplot(happy, hue='label')
# sns.lmplot(x="Economy (GDP per Capita)", y="Happiness Score", data=happy, hue="label", fit_reg=False)
sns.lmplot(x="Economy (GDP per Capita)", y="Happiness Score", data=happy, hue="label", fit_reg=False)
# sns.lmplot(x="Health (Life Expectancy)", y="Happiness Score", data=happy, hue="label", fit_reg=False)
sns.plt.show()
# print happy_result.labels_ 
# sns.pairplot(data, hue="category", diag_kind="kde")

# datset is loaded as a pandas dataframe
# flights = sns.load_dataset("flights")

# sns.set()
# vg = sns.load_dataset(data)
# vg = vg.pivot("month", "year", "passengers")
# g = sns.clustermap(vg)
