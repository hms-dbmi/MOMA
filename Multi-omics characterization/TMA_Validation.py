
import torch
from torch.utils.data import DataLoader

from util.Dataset import load_label, ClusterDataset, load_TMA_data
from util.TransformerMIL import MIL
from util.EMA import EMA
from util.Epoch import TrainEpoch, ValidEpoch

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
    
    model.eval() 

    for i in range(args.kfold):
        print(i)
        
        model.load_state_dict(torch.load('weight/'+task_name+'/R50TransformerMIL_'+task_name+'{}.h5'.format(i+1),map_location='cuda'))
        #model_name = os.path.join(args.save_path, str(i), 'R50TransformerMIL.h5')
        #model.load_state_dict(torch.load(model_name))

        val_epoch = ValidEpoch(model, device = 'cuda', stage = 'TMA External Validation', 
                                positive_count = 0, negative_count = 0)


        val_logs = val_epoch.run(evaluate_loader)

        
#         if(args.evaluate_mode == 'holdout'):
#             break

    
    #draw
#     model.load_state_dict(torch.load('weight/'+task_name+'/R50TransformerMIL_'+task_name+'2.h5',map_location='cuda'))
    
#     val_epoch = ValidEpoch(model, device = 'cuda', stage = 'TMA External Validation', 
#                            positive_count = 0, negative_count = 0)

#     val_logs,ground_truth, model_prediction, auroc = val_epoch.run(evaluate_loader)
    
#     fpr, tpr, _ = metrics.roc_curve(ground_truth,  model_prediction)
    
#     plt.plot(fpr,tpr,label="ROC curve (area = %0.2f)" %(auroc),color='b',lw=2,alpha=0.8)    
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')  
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', alpha=1.0) 
#     plt.legend(loc="lower right")
#     plt.savefig('imgs/TMA/tma_'+task_name+'_roc.jpg')

