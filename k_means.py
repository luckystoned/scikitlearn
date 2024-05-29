import pandas as pd

from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':
    df = pd.read_csv('./data/candy.csv')
    print(df.head())

    x = df.drop('competitorname', axis=1)

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(x)
    print("Total de centroides: ", len(kmeans.cluster_centers_))
    print("="*64)
    print(kmeans.predict(x))

    df['group'] = kmeans.predict(x)

    print(df)
