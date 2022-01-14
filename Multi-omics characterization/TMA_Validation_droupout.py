
import torch
from torch.utils.data import DataLoader

from util.Dataset import load_label, ClusterDataset, load_TMA_data
from util.TransformerMIL import MIL
from util.EMA import EMA
from util.Epoch import TrainEpoch, ValidEpoch, ValidEpochDroupout

from sklearn.model_selection import KFold, train_test_split
import argparse
import os, random
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve

parser = argparse.ArgumentParser("TMA Extenral Validation Setting")

parser.add_argument('--level', type=str, default = 'slide',
                    help = 'Prediction level, slide or patient')
parser.add_argument('--hidden_dim', type = int, default = 512,
                    help = 'patch features dimension')
parser.add_argument('--encoder_layer', type = int, default = 1,
                    help = 'Number of Transformer Encoder layer')
parser.add_argument('--k_sample', type = int, default = 2,
                    help = 'Number of top and bottom cluster to be selected')
parser.add_argument('--tau', type = float, default = 0.7)
parser.add_argument('--save_path', type = str,
                    help = 'Model save path')

parser.add_argument('--label', type = str, default = None,
                    help = 'path to label pickle file')

parser.add_argument('--evaluate_mode', type = str, default='holdout',
                    help='holdout or kfold')
parser.add_argument('--kfold', type = int, default = 5)


# +
if __name__ == '__main__':
    args = parser.parse_args()

    if(args.label == None):
        raise ValueError('label pickle file path cannot be empty')    

    lookup_dic = load_label(args.label)
    
   
    begin=args.label.rfind('/')
    end=args.label.rfind('.')
    task_name=args.label[begin+1:end]
    
    patches_features, cluster_labels = load_TMA_data()
    available_patient_id = list(lookup_dic.keys())
    
    evaluate_cluster = {}
    for k, v in cluster_labels.items():
        if(k in available_patient_id):
            evaluate_cluster[k] = [[] for i in range(10)]
            for p, c in v.items():
                evaluate_cluster[k][c].append(p)

    evaluate_dataset = ClusterDataset(patches_features, evaluate_cluster, lookup_dic)
    evaluate_loader = DataLoader(evaluate_dataset, batch_size = 1, shuffle = False, num_workers = 1, pin_memory = True, drop_last = True)

    model = MIL(hidden_dim = args.hidden_dim, encoder_layer = args.encoder_layer, k_sample = args.k_sample, tau = args.tau)
    
    
    model = EMA(model, 0.999)
    
    model.load_state_dict(torch.load('weight/'+task_name+'/R50TransformerMIL_'+task_name+'2.h5',map_location='cuda')) ##
    
    model.train() #eval

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    
    #draw
    
    for i in range(0,5):
    
        val_epoch = ValidEpochDroupout(model, device = 'cuda', stage = 'TMA External Validation', 
                                       positive_count = 0, negative_count = 0)

        val_logs,ground_truth, model_prediction, auroc = val_epoch.run(evaluate_loader)
        
        fpr, tpr, _ = metrics.roc_curve(ground_truth,  model_prediction)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auroc)
        
        metrics.plot_roc_curve

        plt.plot(fpr,tpr,label="ROC fold "+str(i)+" (area = %0.2f)" %(auroc),lw=1,alpha=0.5)
    
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc) ,
         lw=2, alpha=.8)
        
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  

      
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=1.0) 

plt.legend(loc="lower right")    
plt.savefig('{}_roc.jpg'.format(i+10))    
    
    
#     fpr, tpr, _ = metrics.roc_curve(ground_truth,  model_prediction)
    
#     plt.plot(fpr,tpr,label="ROC curve (area = %0.2f)" %(auroc),color='b',lw=2,alpha=0.8)    
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')  
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=1.0) 
#     plt.legend(loc="lower right")
#     plt.savefig('imgs/TMA/tma_'+task_name+'_roc.jpg')

# -


