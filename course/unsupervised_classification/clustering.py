import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

def run_kmeans(k=4):
    df = pd.read_csv("data_cache/pca.csv")

    model = KMeans(n_clusters=k, n_init="auto")
    df["cluster"] = model.fit_predict(df)

    df.to_csv("data_cache/kmeans.csv", index=False)

    fig = px.scatter(
        df, x="PC1", y="PC2", color=df["cluster"].astype(str),
        title="K-Means Clustering"
    )
    fig.write_html("vignettes/unsupervised/cache/kmeans_plot.html")
