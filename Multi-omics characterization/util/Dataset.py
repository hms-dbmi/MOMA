# -*- coding: utf-8 -*-
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
import joblib
import pickle


# def load_TMA_data():
#     with open(file_to_patch_features, 'rb') as f:  #please input Patch features pickle
#         patch_features = pickle.load(f)

#     #TMA每個slide切下來的patch太少張了，所以每張patch都有去做9次aug，來增加資料量
#     with open('../data/TMA/R50ImageNet_224x224(512_256)_norm_augment_9_feature_dict.pkl', 'rb') as f:
#         patch_features.update(pickle.load(f))

#     with open('../data/TMA/Feature/R50ImageNet_224x224(512_256)_tumorFilter_TumorId_cluster_label_k=10_less30.pkl', 'rb') as f:
#         cluster_label = pickle.load(f)

#     for k, v in cluster_label.items():
#         temp = {}
#         for p, c in v.items():
#             for i in range(9):
#                 temp.update({p + '_aug_{}'.format(i+1) : c})
#         v.update(temp)

#     return patch_features, cluster_label    

def load_data(cancer_type = 'COAD', level = 'slide', use_kather_data = True):
    if(use_kather_data):
        with open(file_to_patch_features_pickle, 'rb') as f:  #please input
            patch_features = pickle.load(f)
        with open(file_to_cluster_pickle, 'rb') as f:  #please input
            cluster_label = pickle.load(f)

        return patch_features, cluster_label 

    if cancer_type == 'COAD':

        patch_features = joblib.load(file_to_patch_features_pickle_COAD)  #please input

        with open(file_to_cluster_pickle, 'rb') as f:  #please input
            cluster_label = pickle.load(f)
            
    elif cancer_type == 'READ':
        
        with open(file_to_patch_features_pickle_READ, 'rb') as f:  #please input
            patch_features = pickle.load(f)

        with open(file_to_cluster_pickle_READ, 'rb') as f:  #please input
            cluster_label = pickle.load(f)

    elif cancer_type == "CRC" :

        patch_features = joblib.load(file_to_patch_features_pickle_COAD)  #please input

        with open(file_to_patch_features_pickle_READ, 'rb') as f:  #please input
            patch_features.update(pickle.load(f))


        with open(file_to_cluster_pickle_COAD, 'rb') as f:  #please input
            cluster_label = pickle.load(f)

        with open(file_to_cluster_pickle_READ, 'rb') as f:  #please input
            cluster_label.update(pickle.load(f))
    else:
        raise ValueError("Not an available cancer type!")

        
    return patch_features, cluster_label

def load_label(path):
    # if(task == 'CNA'):
    #     with open('/data/Labels/{}_{}.pkl'.format(gene, type), 'rb') as f:
    #         dic = pickle.load(f)
    #     return dic
    # elif(task == 'MSI'):
    #     with open('/data/Labels/Kather_MSI.pkl', 'rb') as f:
    #         dic = pickle.load(f)
    #     return dic
    # else:
    #     with open('/data/Labels/TCGA_{}.pkl'.format(task), 'rb') as f:
    #         dic = pickle.load(f)
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic       


def get_available_id(lookup, cluster_label):
    has_label_patient_id = list(lookup.keys())
    available_ids = []
    for p in cluster_label.keys():
        p = p[:12]
        if(p not in available_ids and p in has_label_patient_id):
            available_ids.append(p)
    return available_ids


def drop(cluster, threshold):
    drop = []
    for k, v in cluster.items():
        total = sum([len(c) for c in v])
        if(total < threshold):
            drop.append(k)

    for s_id in drop:
        del cluster[s_id]


def create_cv_data(patients, all_features, cluster_label, train_index, test_index, lookup, level = 'slide'):
    train_patient, test_patient = np.array(patients)[train_index], np.array(patients)[test_index]
    train_cluster = {}
    test_cluster = {}
    
    for k, v in cluster_label.items():
        p_id = k[:12]
        id_ = k[:12] if level == 'patient' else k[:23]
        if(p_id in train_patient):
            train_cluster[id_] = [[] for i in range(10)]
            for p, c in v.items():
                # if(p not in tumor_patch):
                #     continue
                if(p[13] == '1') : continue
                train_cluster[id_][c].append(p)
        elif(p_id in test_patient):
            test_cluster[id_] = [[] for i in range(10)]
            for p, c in v.items():
                # if(p not in tumor_patch):
                #     continue
                test_cluster[id_][c].append(p)    
                
                
    drop(train_cluster, 50)
    drop(test_cluster, 50)
    pos_count = 0    
    for k in train_cluster.keys():
        pos_count += lookup[k[:12]]
    test_pos_count = 0    
    for k in test_cluster.keys():
        test_pos_count += lookup[k[:12]]

    print('Training Postive: %d, Training Negative: %d' %(pos_count, len(train_cluster) - pos_count))
    print('Testing Postive: %d, Testing Negative: %d' %(test_pos_count, len(test_cluster) - test_pos_count))
    
    train_dataset = ClusterDataset(
        features = all_features,
        cluster = train_cluster,
        cls_lookup = lookup,
    )

    test_dataset = ClusterDataset(
        features = all_features,
        cluster = test_cluster,
        cls_lookup = lookup,
    )
    
    return train_dataset, test_dataset, pos_count, len(train_cluster) - pos_count


def wgd_create_cv_data(patients, all_features, cluster_label, train_index, test_index, lookup, level = 'slide'):
    train_patient, test_patient = np.array(patients)[train_index], np.array(patients)[test_index]
    train_cluster = {}
    test_cluster = {}
    
    for k, v in cluster_label.items():
        p_id = k[:15]
        id_ = k[:12] if level == 'patient' else k[:23]
        if(p_id in train_patient):
            train_cluster[id_] = [[] for i in range(10)]
            for p, c in v.items():
                # if(p not in tumor_patch):
                #     continue
                if(p[13] == '1') : continue
                train_cluster[id_][c].append(p)
        elif(p_id in test_patient):
            test_cluster[id_] = [[] for i in range(10)]
            for p, c in v.items():
                # if(p not in tumor_patch):
                #     continue
                if(p[13] == '1') : continue
                test_cluster[id_][c].append(p)    
                
                
    drop(train_cluster, 50)
    drop(test_cluster, 50)
        
    pos_count = 0    
    for k in train_cluster.keys():
        pos_count += lookup[k[:15]]

    test_pos_count = 0    
    for k in test_cluster.keys():
        test_pos_count += lookup[k[:15]]

    print('Training Postive: %d, Training Negative: %d' %(pos_count, len(train_cluster) - pos_count))
    print('Testing Postive: %d, Testing Negative: %d' %(test_pos_count, len(test_cluster) - test_pos_count))

    train_dataset = ClusterDataset(
        features = all_features,
        cluster = train_cluster,
        cls_lookup = lookup,
    )


    test_dataset = ClusterDataset(
        features = all_features,
        cluster = test_cluster,
        cls_lookup = lookup,
    )
    
    return train_dataset, test_dataset, pos_count, len(train_cluster) - pos_count


def msimss_create_cv_data(patients, all_features, cluster_label, train_index, test_index, lookup, level = 'patient'):
    train_patient, test_patient = np.array(patients)[train_index], np.array(patients)[test_index]
    train_cluster = {}
    test_cluster = {}
    
    for k, v in cluster_label.items():
        p_id = k[:12]
        if(p_id in train_patient):
            train_cluster[p_id] = [[] for i in range(10)]
            for p, c in v.items():
#                 if(p not in tumor_patch):
#                     continue
                train_cluster[p_id][c].append(p)
        elif(p_id in test_patient):
            test_cluster[p_id] = [[] for i in range(10)]
            for p, c in v.items():
#                 if(p not in tumor_patch):
#                     continue
                test_cluster[p_id][c].append(p)    
                
                
    drop(train_cluster, 50)
    drop(test_cluster, 50)
        
    pos_count = 0    
    for k in train_cluster.keys():
        pos_count += lookup[k[:12]]
    test_pos_count = 0    
    for k in test_cluster.keys():
        test_pos_count += lookup[k[:12]]

    print('Training Postive: %d, Training Negative: %d' %(pos_count, len(train_cluster) - pos_count))
    print('Testing Postive: %d, Testing Negative: %d' %(test_pos_count, len(test_cluster) - test_pos_count))
            
    train_dataset = ClusterDataset(
        features = all_features,
        cluster = train_cluster,
        cls_lookup = lookup,
    )


    test_dataset = ClusterDataset(
        features = all_features,
        cluster = test_cluster,
        cls_lookup = lookup,
    )
    
    return train_dataset, test_dataset, pos_count, len(train_cluster) - pos_count


class ClusterDataset(Dataset):
    def __init__(self, features, cluster, cls_lookup, score_lookup = None):
        self.patient = list(cluster.keys())
        self.features = features
        self.cluster = cluster
        self.cls_lookup = cls_lookup
        self.score_lookup = score_lookup
    def __getitem__(self, i):
        p_id = self.patient[i]
        cluster = self.cluster[p_id]
        if(isinstance(p_id, str)):
            p_id = p_id[:12]
        data = []
        patch_name = []
        for ind, patch in enumerate(cluster):              
            feature = []
            pn = []
            if(len(patch) == 0):
                continue
            for p in patch:
                feature.append(self.features[p])
                pn.append(p)
            feature = np.array(feature)
            data.append(feature)
            patch_name.append(pn)
        if(self.score_lookup is not None):
            score = self.score_lookup[p_id]
            
            return data, np.array([1- score , score]), self.cls_lookup[p_id]
        else:
            return data, self.cls_lookup[p_id], patch_name
        
        
    def __len__(self):
        return len(self.patient)

    

