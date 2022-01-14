
import torch
from torch.utils.data import DataLoader

from util.Dataset import load_data, load_label, get_available_id, create_cv_data, msimss_create_cv_data, wgd_create_cv_data
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

parser = argparse.ArgumentParser("Experimental Setting")

parser.add_argument('--cancer_type', type = str, default = 'COAD',
                    help = 'Only including COAD, READ and CRC')
parser.add_argument('--level', type=str, default = 'slide',
                    help = 'Prediction level, slide or patient')
parser.add_argument('--hidden_dim', type = int, default = 512,
                    help = 'patch features dimension')
parser.add_argument('--encoder_layer', type = int, default = 1,
                    help = 'Number of Transformer Encoder layer')
parser.add_argument('--k_sample', type = int, default = 2,
                    help = 'Number of top and bottom cluster to be selected')
parser.add_argument('--save_path', type = str,
                    help = 'Model save path')

# parser.add_argument('--task', type = str,
#                     help = 'MSI, CNA, WDG, KRAS, etc.')
# parser.add_argument('--gene', type = str)
# parser.add_argument('--mutation_type', type = str, default = None,
#                     help = 'Deletion or Amplification')

parser.add_argument('--use_kather_data', type = bool, default = True)
parser.add_argument('--label', type = str, default = None,
                    help = 'path to label pickle file')

parser.add_argument('--lr', type = float, default = 3e-4)
parser.add_argument('--epoch', type = int, default = 60)
parser.add_argument('--tau', type = float, default = 0.7)
parser.add_argument('--evaluate_mode', type = str, default='holdout',
                    help='holdout or kfold')
parser.add_argument('--kfold', type = int, default = 5)


# +
if __name__ == '__main__':
    random.seed(1)
    args = parser.parse_args()
    
    # if(args.task == 'CNA'):
    #     if(args.gene == None or args.mutation_type == None ):
    #         raise ValueError('gene and mutation type parameters cannot be empty')

    if(args.label == None):
        raise ValueError('label pickle file path cannot be empty')
    if(args.use_kather_data and args.level == 'slide'):
        raise ValueError('if you want to use kather et al. dataset, you have to set level to patient-level')

    k_fold = KFold(n_splits = args.kfold)
    
    begin=args.label.rfind('/')
    end=args.label.rfind('.')
    task_name=args.label[begin+1:end]
    

    lookup_dic = load_label(args.label)
    patches_features, cluster_labels = load_data(cancer_type = args.cancer_type,
                                                level = args.level, use_kather_data= args.use_kather_data)

    available_patient_id = get_available_id(lookup_dic, cluster_labels)
    available_patient_id = random.sample(available_patient_id, len(available_patient_id))
    
    # if(args.gene == 'MSI'):
    #     cv_data_func = msimss_create_cv_data
    # elif (args.gene == 'WGD'):
    #     cv_data_func = wgd_create_cv_data
    # else:
    cv_data_func = create_cv_data
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for i, (train_index, test_index) in enumerate((k_fold.split(available_patient_id))):
        if(args.evaluate_mode == 'holdout'):
            train_index, val_index = train_test_split(train_index, test_size = 0.25, random_state=42)
        else:
            val_index = test_index
        train_dataset, test_dataset, positive_count, negative_count = cv_data_func(available_patient_id,
                                                                                    patches_features,
                                                                                    cluster_labels,
                                                                                    train_index,
                                                                                    val_index,
                                                                                    lookup_dic,
                                                                                    level = args.level)
            
        train_loader = DataLoader(train_dataset, batch_size = 1, shuffle=True,
                                    num_workers = 4, pin_memory = True, drop_last = False)
        val_loader = DataLoader(test_dataset, batch_size = 1, shuffle=True,
                                    num_workers = 4, pin_memory = True, drop_last = False)

        model = MIL(hidden_dim = args.hidden_dim, encoder_layer = args.encoder_layer, k_sample = args.k_sample, tau = args.tau)
        model = EMA(model, 0.999)
        
        #model.load_state_dict(torch.load('weight/PIK3CA/R50TransformerMIL_PIK3CA{}.h5'.format(i+1),map_location='cuda'))
        model.load_state_dict(torch.load('weight/'+task_name+'/R50TransformerMIL_'+task_name+'{}.h5'.format(i+1),map_location='cuda'))
        
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, weight_decay = 1e-2, momentum = 0.9, nesterov = True) #5e-4
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epoch, eta_min = 0, last_epoch = -1)


#         train_epoch = TrainEpoch(model, device = 'cuda', stage = 'Train', optimizer = optimizer, 
#                                     positive_count = positive_count, negative_count = negative_count)
        val_epoch = ValidEpoch(model, device = 'cuda', stage = 'Valid', 
                                positive_count = positive_count, negative_count = negative_count)


        max_auc = 0
        min_loss = float('inf')

#         model_name = os.path.join(args.save_path, str(i), 'R50TransformerMIL.h5')

#         for epoch_count in range(1, args.epoch + 1) :
#             print('Epoch: {}'.format(epoch_count))
#             train_logs = train_epoch.run(train_loader)
        val_logs,ground_truth, model_prediction, auroc = val_epoch.run(val_loader)
        
        fpr, tpr, _ = metrics.roc_curve(ground_truth,  model_prediction)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auroc)
        
        metrics.plot_roc_curve
        
        #tpr_list.append(np.array(tpr))
        #fpr_list.append(np.array(fpr))
        
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
