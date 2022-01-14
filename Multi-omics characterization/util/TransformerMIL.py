import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from topk import SmoothTop1SVM

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

    


class MIL(nn.Module):
    def __init__(self, hidden_dim = 512, num_class = 2, encoder_layer = 1, k_sample = 2, tau = 0.7):
        super().__init__()
        
        self.k_sample = k_sample
        self.n_classes = num_class
        self.L = hidden_dim
        self.D = hidden_dim
        self.K = 1
        self.subtyping = True
        
        self.instance_loss_fn = SmoothTop1SVM(num_class, tau = tau).cuda()
        
        self.attention_V2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U2 = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights2 = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 2)
        )

        instance_classifiers = [nn.Linear(hidden_dim, 2) for i in range(num_class)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        self.cls_token = nn.Parameter(torch.zeros((1, 1, hidden_dim)))
        self.projector = nn.Linear(2048, hidden_dim)
        self.transformer = Transformer(hidden_dim, encoder_layer, 8, 64, 2048, 0.1)
        self.dropout = nn.Dropout(0.1)
        
        
        
    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device, dtype = torch.long)
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device, dtype = torch.long)
    
    def inst_eval(self, A, h, instance_feature, classifier): 
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        #top_p = torch.index_select(instance_feature, dim=0, index=top_p_ids)
        top_p = [instance_feature[i] for i in top_p_ids]
        top_p = torch.cat(top_p, dim = 1).squeeze(0)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        #top_n = torch.index_select(instance_feature, dim=0, index=top_n_ids)
        top_n = [instance_feature[i] for i in top_n_ids]
        top_n = torch.cat(top_n, dim = 1).squeeze(0)
        p_targets = self.create_positive_targets(len(top_p), device)
        n_targets = self.create_negative_targets(len(top_n), device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets
    
    #instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, instance_feature, classifier):
        device=h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, len(A))[1][-1]
        top_p = [instance_feature[i] for i in top_p_ids]
        top_p = torch.cat(top_p, dim = 1).squeeze(0)
        p_targets = self.create_negative_targets(len(top_p), device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets    
    
    def forward(self, xs, label):
        H = []
        instance_feature = []
        for x in xs:
            #x = x.permute(1,0, 2)
            x = self.projector(x)
            x = torch.cat((self.cls_token, x), dim = 1)
            x = self.dropout(x)
            rep = self.transformer(x)
            H.append(rep[:, 0])
            instance_feature.append(rep[:, 1:])
            
        H = torch.cat(H)
        A_V = self.attention_V2(H)  # NxD
        A_U = self.attention_U2(H)  # NxD
        A = self.attention_weights2(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        
        
        total_inst_loss = 0.0
        all_preds = []
        all_targets = []
        inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label
        for i in range(len(self.instance_classifiers)):
            inst_label = inst_labels[i].item()
            classifier = self.instance_classifiers[i]
            if inst_label == 1: #in-the-class:
                instance_loss, preds, targets = self.inst_eval(A, H, instance_feature, classifier)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
            else: #out-of-the-class
                if self.subtyping:
                    instance_loss, preds, targets = self.inst_eval_out(A, H, instance_feature, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:
                    continue
            total_inst_loss += instance_loss        
        
        
        total_inst_loss /= len(self.instance_classifiers)
        
        
        
        M = torch.mm(A, H)  # KxL        
        logit = self.classifier(M)
    
        
        return logit, total_inst_loss, A.detach()
