import seaborn as sns
import pandas as pd

sns.set()

df = pd.read_csv('./datasets/happy2016.csv')
sns.pairplot(df, hue="Region")
sns.plt.show()