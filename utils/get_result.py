import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas_profiling
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,confusion_matrix,accuracy_score,classification_report,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sklearn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

def get_result(y_test,anomaly_data):
    normal_data = anomaly_data[anomaly_data['binary_labels']==0]
    anormal_data = anomaly_data[anomaly_data['binary_labels']==1]
    scores=anomaly_data["recon_score"].values
    normal_data_score = normal_data['recon_score'].to_numpy()
    anormal_data_score = anormal_data['recon_score'].to_numpy()
    bins = np.linspace(-0.000, 1.000, num = 100)
    kwargs = dict(histtype='stepfilled', alpha=0.3, bins=bins)
    
    n_n, n_bins = np.histogram(normal_data_score, bins = bins)
    an_n, an_bins = np.histogram(anormal_data_score, bins = bins);
    
    n_density = n_n/np.sum(n_n)
    an_density = an_n/np.sum(an_n)
    plt.plot(bins[0:99], n_density, label = 'normal')
    plt.plot(bins[0:99], an_density, label = 'anormal')
    plt.legend();
    plt.show()
    
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lime', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    optimal_threshold_idx = np.argmax(tpr-fpr)
    optimal_threshold = thresholds[optimal_threshold_idx]
    print(optimal_threshold)
    
    thresh = optimal_threshold
    pred_labels = (scores > thresh).astype(int)
    
    AUROC = roc_auc_score(y_test,pred_labels)
    AUPRC = average_precision_score(y_test,pred_labels)
    recall = recall_score(y_test,  pred_labels)
    precision = precision_score(y_test,pred_labels)
    f1_score_=f1_score(y_true=y_test, y_pred =pred_labels)
    
    results = confusion_matrix(y_test, pred_labels)
    ae_acc = accuracy_score(y_test, pred_labels)
    
    print(ae_acc, precision,recall,AUROC,AUPRC,f1_score_)
    
    print ('\nConfusion Matrix: ')
    
    cm=results
    target_names=['Normal','Anomaly']
    title='Confusion matrix'
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    print("acc:".rjust(10),ae_acc)
    print("precision:".rjust(10),precision)
    print("recall:".rjust(10),recall)
    print("AUROC:".rjust(10),AUROC)
    print("AUPRC:".rjust(10),AUPRC)
    print("f1_score_:".rjust(10),f1_score_)
    