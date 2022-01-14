import torch
import torch.nn.functional as F

from tqdm import tqdm
from .meter import AverageValueMeter
import sys


from sklearn.metrics import roc_auc_score, average_precision_score
class Epoch:
    def __init__(self, model, device = 'cuda', stage = 'Train', optimizer = None, positive_count = 0, negative_count = 0):
        self.model = model
        self.device = device
        self.stage = stage
        self.optimizer = optimizer
        self.to_device()

        self.positive_count = positive_count
        self.negative_count = negative_count
        self.accumulation_steps = 32

    def to_device(self):
        self.model.to(self.device)


    def on_epoch_start(self):
        pass

    def batch_update(self):
        pass

    def run(self, dataloder):
        self.on_epoch_start()

        loss_meter = AverageValueMeter()
        logs = {}

        with tqdm(dataloder, desc = self.stage, file = sys.stdout) as iterator:
            for i, (x, y, patch_names) in enumerate(iterator):
                x = [xi.to(self.device) for xi in x]
                y = y.to(self.device)
                
                loss, pred, weights = self.batch_update(x, y)
                if((i+1)%self.accumulation_steps == 0):
                    self.optimizer.step()
                    loss = loss.cpu().detach().numpy()
                    loss_meter.add(loss)
                    logs.update({'loss' : loss})

                    iterator.set_postfix_str('Loss:'+str(logs['loss']))

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, device = 'cuda', stage = 'Train', optimizer = None, positive_count = 0, negative_count = 0):
        super().__init__(
            model,
            device,
            stage,
            optimizer,
            positive_count,
            negative_count
        )

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        pred, instance_loss, cluster_attention_weight = self.model(x, y)
        bag_loss = F.cross_entropy(pred, y, weight = torch.tensor([1/self.negative_count, 1/self.positive_count]).to(self.device).float())
        total_loss = bag_loss + instance_loss
        total_loss = total_loss / self.accumulation_steps
        total_loss.backward()

        return total_loss, pred, cluster_attention_weight


class ValidEpoch(Epoch):
    def __init__(self, model, device = 'cuda', stage = 'Train', optimizer = None, positive_count = 0, negative_count = 0):
        super().__init__(
            model,
            device,
            stage,
            optimizer,
            positive_count,
            negative_count
        )
    
    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            pred, instance_loss, cluster_attention_weight = self.model(x, y)
            bag_loss = F.cross_entropy(pred, y)
            total_loss = bag_loss

        return total_loss, pred, cluster_attention_weight


    def run(self, dataloder):
        
        
        self.on_epoch_start()

        loss_meter = AverageValueMeter()
        logs = {}

        ground_truth = []
        model_prediction = []

        with tqdm(dataloder, desc = self.stage, file = sys.stdout) as iterator:
            for x, y, patch_names in iterator:
                x = [xi.to(self.device) for xi in x]
                y = y.to(self.device)
                
                loss, pred, weights = self.batch_update(x, y)
                loss = loss.cpu().detach().numpy()
                loss_meter.add(loss)
                logs.update({'loss' : loss})

                iterator.set_postfix_str('Loss:'+str(logs['loss']))

                #store each bag label and prediction
                y = y.detach().cpu().numpy()
                output = torch.softmax(pred, dim = 1)[:, 1].detach().cpu().numpy()
                for p, yi in zip(output, y):
                    model_prediction.append(p)
                    ground_truth.append(yi)

        auroc = roc_auc_score(ground_truth, model_prediction)
        aupr = average_precision_score(ground_truth, model_prediction)

        print('AUC: %0.2f, AUPR: %0.2f' %(auroc, aupr))







        logs.update({'AUC':aupr, 'AUPR':aupr})

        return logs, ground_truth, model_prediction, auroc


class ValidEpochDroupout(Epoch):
    def __init__(self, model, device = 'cuda', stage = 'Train', optimizer = None, positive_count = 0, negative_count = 0):
        super().__init__(
            model,
            device,
            stage,
            optimizer,
            positive_count,
            negative_count
        )
    
    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            pred, instance_loss, cluster_attention_weight = self.model(x, y)
            bag_loss = F.cross_entropy(pred, y)
            total_loss = bag_loss

        return total_loss, pred, cluster_attention_weight


    def run(self, dataloder):
        
        
        #self.on_epoch_start()

        loss_meter = AverageValueMeter()
        logs = {}

        ground_truth = []
        model_prediction = []

        with tqdm(dataloder, desc = self.stage, file = sys.stdout) as iterator:
            for x, y, patch_names in iterator:
                x = [xi.to(self.device) for xi in x]
                y = y.to(self.device)
                
                loss, pred, weights = self.batch_update(x, y)
                loss = loss.cpu().detach().numpy()
                loss_meter.add(loss)
                logs.update({'loss' : loss})

                iterator.set_postfix_str('Loss:'+str(logs['loss']))

                #store each bag label and prediction
                y = y.detach().cpu().numpy()
                output = torch.softmax(pred, dim = 1)[:, 1].detach().cpu().numpy()
                for p, yi in zip(output, y):
                    model_prediction.append(p)
                    ground_truth.append(yi)

        auroc = roc_auc_score(ground_truth, model_prediction)
        aupr = average_precision_score(ground_truth, model_prediction)

        print('AUC: %0.2f, AUPR: %0.2f' %(auroc, aupr))







        logs.update({'AUC':aupr, 'AUPR':aupr})

        return logs, ground_truth, model_prediction, auroc
