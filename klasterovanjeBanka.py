import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import DBSCAN


df = pd.read_csv('./bank.csv')
test = pd.read_csv("bank_output_file.csv")
klaster = np.arange(1, 6, 1)
greske = np.arange(0, 5, 1, dtype=np.int64)
print(df)
#radimo encoding da mi string bio zapravo broj
#ovaj daje nekim a prednost, zato je bolji
#OneHotEncoding
''''
lenc = LabelEncoder()
df['id'] = lenc.fit_transform(df['id'])
df['sex'] = lenc.fit_transform(df['sex'])
df['region'] = lenc.fit_transform(df['region'])
df['married'] = lenc.fit_transform(df['married'])
df['car'] = lenc.fit_transform(df['car'])
df['save_act'] = lenc.fit_transform(df['save_act'])
df['current_act'] = lenc.fit_transform(df['current_act'])
df['mortgage'] = lenc.fit_transform(df['mortgage'])
df['pep'] = lenc.fit_transform(df['pep'])

train = df.drop("id", axis=1)
'''
#sredili smo string kolone
train = pd.get_dummies(df, columns=['id', 'sex', 'region', 'married', 'car', 'save_act', 'current_act', 'mortgage', 'pep'], drop_first=True)
#uradimo standardizaciju
''''
sc = StandardScaler()
sc.fit(train)
train[train.columns] = sc.fit_transform(train[train.columns])
print(train)
'''
''''
for i in range(0, 5):
    greske[i] = KMeans(n_clusters=klaster[i]).fit(train).inertia_

plt.plot(klaster, greske)
plt.show()
'''

plt.boxplot(train['income'])
plt.show()

plt.boxplot(train['children'])
plt.show()
dbscan = DBSCAN(eps=590, min_samples=8)
dbscan.fit(train)
labels = dbscan.fit_predict(train)

ari = adjusted_rand_score(test['cluster'], labels)
print(ari)
kmeans = KMeans(n_clusters=2, random_state=420)
labels = kmeans.fit_predict(train)
print(test['cluster'])
ari = adjusted_rand_score(test['cluster'], labels)
print(ari)
#dbscan = DBSCAN(eps=0.02, min_samples=5)
#dbscan.fit(X)
#labels = dbscan.labels_


