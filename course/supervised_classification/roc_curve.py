import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from course.utils import find_project_root


def _get_roc_results(y_test_path, y_pred_prob_path):
    """
    Compute ROC curves and AUC values for each class.
    Works for multiclass predictions where y_pred_prob CSV has one column per class.
    """
    y_test = pd.read_csv(y_test_path)['built_age']
    y_pred_prob_df = pd.read_csv(y_pred_prob_path)

 
    classes = [col.replace('prob_', '') for col in y_pred_prob_df.columns]


    y_test_bin = label_binarize(y_test, classes=classes)

    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    for i, cls in enumerate(classes):
        fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_pred_prob_df.iloc[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_dict[cls] = fpr
        tpr_dict[cls] = tpr
        auc_dict[cls] = roc_auc

    return {'fpr': fpr_dict, 'tpr': tpr_dict, 'roc_auc': auc_dict}


def plot_roc_curve():
    """
    Plot ROC curves for LDA and QDA models.
    Saves output to data_cache/vignettes/supervised_classification/roc.html
    """
    base_dir = find_project_root()


    y_test_path = base_dir / 'data_cache' / 'energy_y_test.csv'
    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'lda_y_pred_prob.csv'
    lda_results = _get_roc_results(y_test_path, y_pred_prob_path)


    y_pred_prob_path = base_dir / 'data_cache' / 'models' / 'qda_y_pred_prob.csv'
    qda_results = _get_roc_results(y_test_path, y_pred_prob_path)


    fig = _plot_roc_curve(lda_results, qda_results)


    outpath = base_dir / 'data_cache' / 'vignettes' / 'supervised_classification' / 'roc.html'
    fig.write_html(outpath)


def _plot_roc_curve(lda_roc, qda_roc):
    """
    Helper function to generate Plotly ROC figure for LDA and QDA.
    """
    fig = go.Figure()


    for cls, auc_val in lda_roc['roc_auc'].items():
        fig.add_trace(go.Scatter(
            x=lda_roc['fpr'][cls],
            y=lda_roc['tpr'][cls],
            mode='lines',
            name=f'LDA {cls} (AUC={auc_val:.2f})'
        ))

    for cls, auc_val in qda_roc['roc_auc'].items():
        fig.add_trace(go.Scatter(
            x=qda_roc['fpr'][cls],
            y=qda_roc['tpr'][cls],
            mode='lines',
            name=f'QDA {cls} (AUC={auc_val:.2f})'
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    return fig
