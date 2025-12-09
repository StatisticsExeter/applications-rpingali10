import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from course.utils import find_project_root


def _get_roc_results(y_test_path, y_pred_prob_path):
    """
    Compute ROC curve and AUC from test labels and predicted probabilities.
    Supports both binary and multi-class classification.
    """

    y_test = pd.read_csv(y_test_path)['built_age']
    y_pred_prob_df = pd.read_csv(y_pred_prob_path)


    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)


    if y_pred_prob_df.shape[1] == 1:
        y_pred_prob = y_pred_prob_df.iloc[:, 0]
        fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_prob)
        roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr = {}, {}
        roc_auc = {}
        for i, class_label in enumerate(lb.classes_):
            y_prob = y_pred_prob_df.iloc[:, i]
            fpr[class_label], tpr[class_label], _ = roc_curve(y_test_bin[:, i], y_prob)
            roc_auc[class_label] = auc(fpr[class_label], tpr[class_label])
        
        first_class = lb.classes_[0]
        fpr, tpr, roc_auc = fpr[first_class], tpr[first_class], roc_auc[first_class]

    return {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}


def plot_roc_curve():
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lda_roc['fpr'], y=lda_roc['tpr'],
                             mode='lines',
                             name=f'ROC curve from LDA (AUC = {lda_roc["roc_auc"]:.2f})'))
    fig.add_trace(go.Scatter(x=qda_roc['fpr'], y=qda_roc['tpr'],
                             mode='lines',
                             name=f'ROC curve from QDA (AUC = {qda_roc["roc_auc"]:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                             mode='lines',
                             name='Random', line=dict(dash='dash')))
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    return fig
