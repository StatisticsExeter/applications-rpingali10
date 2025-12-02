import pandas as pd
from pathlib import Path

# Existing imports
from course.unsupervised_classification.visual_eda import (
    summary_stats,
    generate_raw_boxplot,
    generate_scaled_boxplot,
    generate_scatterplot
)

# New imports for added tasks
from course.unsupervised_classification.dimension_reduction import run_pca
from course.unsupervised_classification.clustering import run_kmeans


# ----------------------------
#  Folder checks
# ----------------------------

def task_check_cache():
    def check_cache():
        """Ensure cache folder exists"""
        models_path = Path("data_cache/models")
        models_path.mkdir(parents=True, exist_ok=True)
    return {
        'actions': [check_cache]
    }


def task_check_vignettes():
    def check_vignettes():
        """Ensure vignette cache folder exists"""
        vignettes_path = Path("vignettes/unsupervised/cache")
        vignettes_path.mkdir(parents=True, exist_ok=True)
    return {
        'actions': [check_vignettes]
    }


# ----------------------------
#  Load Data
# ----------------------------

def task_load_data():
    def load_data():
        df = pd.read_csv('data/olive_oil.csv')
        df_reduced = df.iloc[:, 3:]   # drop first three columns
        df_reduced.to_csv('data_cache/unsupervised.csv', index=False)

    return {
        'actions': [load_data],
        'file_dep': ['data/olive_oil.csv'],
        'targets': ['data_cache/unsupervised.csv'],
    }


# ----------------------------
#  EDA tasks
# ----------------------------

def task_summary_stats():
    return {
        'actions': [summary_stats],
        'file_dep': [
            'data_cache/unsupervised.csv',
            'course/unsupervised_classification/visual_eda.py'
        ],
        'targets': ['vignettes/unsupervised/cache/olive_oil_summary.html'],
    }


def task_plot_raw_boxplot():
    return {
        'actions': [generate_raw_boxplot],
        'file_dep': [
            'data_cache/unsupervised.csv',
            'course/unsupervised_classification/visual_eda.py'
        ],
        'targets': ['vignettes/unsupervised/cache/raw_boxplot.html'],
        'clean': True,
    }


def task_plot_scaled_boxplot():
    return {
        'actions': [generate_scaled_boxplot],
        'file_dep': [
            'data_cache/unsupervised.csv',
            'course/unsupervised_classification/visual_eda.py'
        ],
        'targets': ['vignettes/unsupervised/cache/scaled_boxplot.html'],
        'clean': True,
    }


def task_plot_scatterplot():
    return {
        'actions': [generate_scatterplot],
        'file_dep': [
            'data_cache/unsupervised.csv',
            'course/unsupervised_classification/visual_eda.py'
        ],
        'targets': ['vignettes/unsupervised/cache/scatterplot.html'],
        'clean': True,
    }


# ----------------------------
#  NEW TASK: PCA (Dimension Reduction)
# ----------------------------

def task_pca():
    return {
        'actions': [run_pca],
        'file_dep': [
            'data_cache/unsupervised.csv',
            'course/unsupervised_classification/dimension_reduction.py',
        ],
        'targets': [
            'data_cache/pca.csv',
            'vignettes/unsupervised/cache/pca_plot.html',
        ],
        'clean': True,
    }


# ----------------------------
#  NEW TASK: K-Means Clustering
# ----------------------------

def task_kmeans():
    return {
        'actions': [run_kmeans],
        'file_dep': [
            'data_cache/pca.csv',
            'course/unsupervised_classification/clustering.py',
        ],
        'targets': [
            'data_cache/kmeans.csv',
            'vignettes/unsupervised/cache/kmeans_plot.html',
        ],
        'clean': True,
    }


# ----------------------------
#  Render Quarto
# ----------------------------

def task_render_quarto():
    return {
        'actions': ['quarto render vignettes/unsupervised/ict_unsupervised.qmd'],
        'file_dep': ['vignettes/unsupervised/ict_unsupervised.qmd'],
        'targets': ['vignettes/unsupervised/ict_unsupervised.html'],
        'clean': True,
    }
