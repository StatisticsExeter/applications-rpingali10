import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from pathlib import Path


def run_pca():
    print(">>> RUNNING PCA FUNCTION <<<")
    base_cache = Path("data_cache")
    vignette_cache = Path("vignettes") / "unsupervised" / "cache"

    # Ensure cache folders exist
    base_cache.mkdir(parents=True, exist_ok=True)
    vignette_cache.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(base_cache / "unsupervised.csv")
    print("Input data shape:", df.shape)

    # Scale and run PCA
    X = StandardScaler().fit_transform(df)
    pca = PCA(n_components=2)
    comps = pca.fit_transform(X)
    out = pd.DataFrame(comps, columns=["PC1", "PC2"])

    # Save CSV
    out.to_csv(base_cache / "pca.csv", index=False)
    print("PCA CSV saved to", base_cache / "pca.csv")

    # Save plot
    fig = px.scatter(out, x="PC1", y="PC2", title="PCA (2 components)")
    fig.write_html(vignette_cache / "pca_plot.html")
    print("PCA HTML plot saved to", vignette_cache / "pca_plot.html")
