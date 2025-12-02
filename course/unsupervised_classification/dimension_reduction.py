import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

def run_pca():
    df = pd.read_csv("data_cache/unsupervised.csv")
    X = StandardScaler().fit_transform(df)

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)

    out = pd.DataFrame(comps, columns=["PC1", "PC2"])
    out.to_csv("data_cache/pca.csv", index=False)

    fig = px.scatter(out, x="PC1", y="PC2", title="PCA (2 components)")
    fig.write_html("vignettes/unsupervised/cache/pca_plot.html")
