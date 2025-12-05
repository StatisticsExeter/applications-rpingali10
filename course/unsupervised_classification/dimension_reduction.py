import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from pathlib import Path

def run_pca():
    print(">>> RUNNING PCA FUNCTION <<<")
    
    # Ensure cache folders exist
    Path("data_cache").mkdir(parents=True, exist_ok=True)
    Path("vignettes/unsupervised/cache").mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv("data_cache/unsupervised.csv")
    print("Input data shape:", df.shape)

    # Scale and run PCA
    X = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    out = pd.DataFrame(comps, columns=["PC1", "PC2"])

    # Save CSV
    out.to_csv("data_cache/pca.csv", index=False)
    print("PCA CSV saved to data_cache/pca.csv")

    # Save plot
    fig = px.scatter(out, x="PC1", y="PC2", title="PCA (2 components)")
    fig.write_html("vignettes/unsupervised/cache/pca_plot.html")
    print("PCA HTML plot saved to vignettes/unsupervised/cache/pca_plot.html")

