#%% import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import itertools
import copy
from tqdm import trange, tnrange

#%% import class     
class RBM(nn.Module):
    def __init__(self, Nv, Nh,k):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.zeros(Nv, Nh))      
        nn.init.normal_(self.W, mean=0, std=0.01)          #gaussian with std 0.01
        self.a = nn.Parameter(torch.zeros(Nv))            #v biases to 0 bc for Z2 symm
        self.b = nn.Parameter(torch.zeros(Nh))            #hidden biases to 0
        self.Nv = Nv
        self.Nh = Nh
        self.k = k
        
    def get_prob_h(self, v):
        return torch.sigmoid(torch.matmul(v, self.W) + self.b)

    def get_prob_v(self, h):
        return torch.sigmoid(torch.matmul(h, self.W.t()) + self.a)

    def sample_h(self, v):
        return self.get_prob_h(v).bernoulli()

    def sample_v(self, h):
        return self.get_prob_v(h).bernoulli()

    def get_grads(self, v_data, h_data, v_rec, h_rec):
        vh_data = torch.einsum('bi,bj->bij', v_data, h_data)
        vh_rec = torch.einsum('bi,bj->bij', v_rec, h_rec)
        # minus sign necessary as we want to maximize loglikelihood
        self.W.grad = -(vh_data.mean(dim=0) - vh_rec.mean(dim=0))
        self.a.grad = -(v_data.mean(dim=0) - v_rec.mean(dim=0))
        self.b.grad = -(h_data.mean(dim=0) - h_rec.mean(dim=0))
        
    def Gibbs_sampling(self, v_data):
        h_data = self.sample_h(v_data)
        v_rec = self.sample_v(h_data)
        for _ in range(self.k-1):
            h_rec = self.sample_h(v_rec)
            v_rec = self.sample_v(h_rec)
        prob_h_rec = self.get_prob_h(v_rec)  #collect hidden prob for negative phase
        h_data = self.sample_h(v_data)       #collect hidden for positive phase
        return v_data, h_data, v_rec, prob_h_rec
    
    # COMPUTE ENERGY FOR EACH ENTRY IN BATCH
    # suppose to have a batch of N vis and M hid, it returns a matrix NxM
    def energy(self, v, h):
        v_term = torch.matmul(v, self.a)
        h_term = torch.matmul(h, self.b).t()
        mixed_term = torch.matmul(torch.matmul(v, self.W), h.t())
        E = -(h_term + v_term + mixed_term)
        return E

    # COMPUTE FREE ENERGY FOR EACH ENTRY IN BATCH
    def energy_v(self, v):
        #its the free energy for each v
        v_term = torch.matmul(v, self.a)
        vw_b = torch.matmul(v, self.W) + self.b
        log_term =  torch.log(1 + torch.exp(vw_b)).sum(dim = 1) 
        return - (v_term + log_term)   
    
    # COMPUTE ENERGY FOR A SINGLE CONFIGURATIONS
    def en(self, v, h):
        v_term = torch.dot(v, self.a)
        h_term = torch.dot(h, self.b)
        mixed_term = torch.dot(v, torch.mv(self.W,h))
        E = -(h_term + v_term + mixed_term)
        return E

    # COMPUTE FREE ENERGY FOR A SINGLE CONFIGURATIONS
    def en_v(self, v):
        v_term = torch.dot(v, self.a)
        vw_b = torch.mv(self.W.t(), v) + self.b
        log_term = torch.log(1 + torch.exp(vw_b)).sum() 
        return - (v_term + log_term)
   
    #functions useful for AIS
    def beta_forward_base(self, v, beta):
        h = torch.sigmoid(beta*torch.matmul(v, self.W) + self.b).bernoulli()
        v = torch.sigmoid(beta*torch.matmul(h, self.W.t()) + self.a).bernoulli()
        return v
    
    def ais_energy_v(self, v, beta):
        v_term = torch.matmul(v, self.a)
        vw_b = torch.matmul(v, self.W)*beta + self.b
        log_term =  torch.log(1 + torch.exp(vw_b)).sum(dim = 1) 
        return - (v_term + log_term)  
    
#training function of RBM
def Train(rbm, train_loader, train_op, device = 'cpu', n_epochs=10, save_model_at=[]):
    save_model_at = [1] + np.array(save_model_at).tolist() + [n_epochs]
    epoch_error_ = []
    model_ = []
    model_.append([0,copy.deepcopy(rbm.state_dict())])
    rbm.to(device)
    for epoch in trange(1, n_epochs+1):
        batch_error_ = []
        for i, v_data in enumerate(train_loader):
            v_data = v_data.float().to(device)
            #GIBBS SAMPLING
            v_data, h_data, v_rec, prob_h_rec = rbm.Gibbs_sampling(v_data)
            #COMPUTE GRADS
            train_op.zero_grad()
            rbm.get_grads(v_data, h_data, v_rec, prob_h_rec)    
            #UPDATE WEIGHTS
            train_op.step()
            #COMPUTE ERROR
            batch_error = torch.mean((v_data - v_rec)**2)
            batch_error_.append(batch_error)
        epoch_error = torch.Tensor(batch_error_).mean()
        epoch_error_.append([epoch, epoch_error])
        if epoch in save_model_at: 
            x = copy.deepcopy(rbm.state_dict())
            model_.append([epoch, x])
    return epoch_error_ , model_


#ANNEALED IMPORTANCE SAMPLING with base rate RBM as starting distribution
def ais_base(rbm, device = 'cpu', n_betas = 1000, M = 200, cdk=5):
    nvis= rbm.a.size()[0]
    nhid = rbm.b.size()[0]
    #define array of intermidiate betas
    
    betas = np.linspace(0, 1, n_betas)
    #extract uniformly and independently M visible configs
    vis = ((1/(1+torch.exp(-rbm.a)))*torch.ones([M, nvis])).bernoulli()    #M,nvis
    logw =  rbm.ais_energy_v(vis, betas[0])               #denom #M
    for idx in range(1,n_betas-1):
        logw -= rbm.ais_energy_v(vis, betas[idx])   #num
        for _ in range(cdk):
            vis = rbm.beta_forward_base(vis, betas[idx])
        logw += rbm.ais_energy_v(vis, betas[idx])   #denom
    logw -= rbm.energy_v(vis)                       #num
    logz0 = torch.log((1+torch.exp(rbm.a))).sum() + torch.log(1+torch.exp(rbm.b)).sum()
    logz1 = torch.logsumexp(logw, 0) - torch.log(torch.Tensor([M])) 
    logz1 = logz1 + logz0
    return logz1
