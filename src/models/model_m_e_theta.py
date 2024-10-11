from src.models.common import *
from src.models.layers_alt import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from torch.optim.lr_scheduler import LinearLR

class m_e_theta_daily(Transformer_core):
    def __init__(self, h=60,
                 proj_len = 5,
                 tgt_size: int = 3,
                 n_features_stat=44, 
                 multi_input_size=27, 
                 hidden_size: int = 128, 
                 n_head: int = 4,
                 learning_rate:float = 0.001,
                 attn_dropout: float = 0.1, 
                 dropout: float = 0.1, 
                 n_categorical_stat:int = None, 
                 embedding_size_stat:list = None, 
                 n_categorical:int = None, 
                 embedding_size:list = None,
                 last_nn = [128],
                 serie_size:int = 1,
                 n_att_layers:int=1,
                 n_static:int=3,
                 cdf:bool=True,
                 padding:int=128,
                 treatment_max : int = 2,
                 n_block:int = 1,
                 kernel_size:int=7,
                 weight_decay:float=1e-4
                 ):
        """This model gathers all 3 models needed to perform the orthogonal learning of the treatment effect. This class is a child of the TFT_core class. We have 
        added 3 MLP at the end of the TFT. Each MLP focus on one models, respectively m_0, e_0 and theta. We first train this model to get good estimation of m_0 and e_0.
        We then freeze all the layers of the model excepts the ones corresponding to the MLP of theta.

        """
        super().__init__(h,  tgt_size, n_features_stat, multi_input_size, hidden_size,
                         n_head, attn_dropout, dropout, learning_rate, n_categorical_stat, embedding_size_stat, n_categorical,
                         embedding_size, serie_size,n_att_layers,n_static,padding,n_block)
        

        self.loss_regression = nn.MSELoss()
        self.is_cdf = cdf
        self.treatment_max = treatment_max
        self.norm_function_e_0 = nn.Softmax(dim = 2)
        self.training_m_e = True
        self.proj_len = proj_len #length of prediction
        ## We set the loss for e_0 regarding the encoding of the treatment
        if self.is_cdf:
            self.loss_classification = nn.functional.binary_cross_entropy
        else:
            self.loss_classification = nn.functional.cross_entropy
        self.eps = 1e-7
        

        self.weight_decay = weight_decay
        ## We create all three models
        self.m = Transformer_core(h,  tgt_size, n_features_stat, multi_input_size+1, hidden_size,
                         n_head, attn_dropout, dropout, learning_rate, n_categorical_stat, embedding_size_stat, n_categorical,
                         embedding_size, serie_size,n_att_layers,n_static,padding,n_block,kernel_size=kernel_size)
        self.e = Transformer_core(h,  tgt_size, n_features_stat, multi_input_size+1, hidden_size,
                         n_head, attn_dropout, dropout, learning_rate, n_categorical_stat, embedding_size_stat, n_categorical,
                         embedding_size, serie_size,n_att_layers,n_static,padding,n_block,kernel_size=kernel_size)
        self.theta_att = Transformer_core(h,  tgt_size, n_features_stat, multi_input_size+1, hidden_size,
                         n_head, attn_dropout, dropout, learning_rate, n_categorical_stat, embedding_size_stat, n_categorical,
                         embedding_size, serie_size,n_att_layers,n_static,padding,n_block,kernel_size=kernel_size)
        self.m_0 = create_sequential(last_nn, hidden_size, tgt_size)
        self.e_0 = create_sequential(last_nn, hidden_size, 2**treatment_max)
        self.theta = create_sequential(last_nn, hidden_size, 2**treatment_max)

        
        self.configure_optimizers()
        self.save_hyperparameters()

    def forward(self, window_batch,tau):
        ## Training m_0, e_0
        if self.training_m_e:
            #Compute m_0
            m_0 = self.m_0(self.m.get_Z(window_batch,tau))
            #compute e_0 vector 
            e_0 = self.e_0(self.e.get_Z(window_batch,tau))
            if self.is_cdf:
                e_0 = self.norm_function_e_0(e_0)
                e_0 = 1 - torch.cumsum(e_0,dim = -1)
                e_0 = e_0 - self.eps
                e_0 = F.relu(e_0)
            theta = torch.ones_like(e_0)
        
        ## Training theta
        elif (self.training and (not self.training_m_e)):
            with torch.no_grad():
                #Compute m_0
                m_0 = self.m_0(self.m.get_Z(window_batch,tau))
                #compute e_0 vector 
                e_0 = self.e_0(self.e.get_Z(window_batch,tau))
                if self.is_cdf:
                    e_0 = self.norm_function_e_0(e_0)
                    e_0 = 1 - torch.cumsum(e_0,dim = -1)
                    e_0 = e_0 - self.eps
                    e_0 = F.relu(e_0)
                else:
                    e_0 = self.norm_function_e_0(e_0)
            Z_theta = self.theta_att.get_Z(window_batch,tau)
            theta = self.theta(Z_theta)
        
        ## Evaluating  
        else:
            with torch.no_grad():
                m_0 = self.m_0(self.m.get_Z(window_batch,tau))
                #compute e_0 vector 
                e_0 = self.e_0(self.e.get_Z(window_batch,tau))
                if self.is_cdf:
                    e_0 = self.norm_function_e_0(e_0)
                    e_0 = 1 - torch.cumsum(e_0,dim = -1)
                    e_0 = e_0 - self.eps
                    e_0 = F.relu(e_0)

                    # in best version: above is commented and replaced by e_0 = F.sigmoid(e_0)
                else:
                    e_0 = self.norm_function_e_0(e_0)
                theta = self.theta(self.theta_att.get_Z(window_batch,tau))
        return m_0, e_0, theta
    

    def forecast(self, batch):
        """Forecast function to return the debiais estimation outcome

        Args:
            batch (dict): dictionnary containing all relevent informations of the batch
            indexes (torch.tensor): Tensor containing tau values

        Returns:
            torch.tensor: Tensor of the outcome
        """
        vitals = batch['vitals'].float()
        static_features = batch['static_features'].float()
        treatments = batch['current_treatments'].float()
        position = torch.arange(batch['vitals'].shape[1])
        position = position.repeat(batch['vitals'].shape[0],1,1)
        position = torch.permute(position,(0,2,1)).to(batch['vitals'].device)
        
        Y = batch["outputs"].float()
        active_entries = torch.zeros_like(Y)
        li_insample_y = batch["outputs"].clone().float()
        for i,tau in enumerate(batch["future_past_split"].int()):
            li_insample_y[i,tau:] = 0
            active_entries[i,tau:tau+self.proj_len] = 1
            vitals[i,tau:] = 0
        index = batch["future_past_split"].int()
        

        temporal = torch.concat([vitals,position,treatments],dim=-1)
        treatment = torch.zeros_like(Y)
        for k in range(self.treatment_max):
            treatment += temporal[:,:,-1-k].clone().unsqueeze(-1)*(2**k)
            for i,tau in enumerate(index):
                temporal[i,tau:,-1-k] = -1
        # Encapsulating inputs
        windows = {}
        windows["insample_y"] = li_insample_y
        windows["multivariate_exog"] = temporal
        windows["stat_exog"] = static_features
        
        
        m_0, e_0, theta = self.forward(windows,index)
        T = treatment[:,:,0].long()
        T = F.one_hot(T, 2**self.treatment_max).float()
        if self.is_cdf:
            # in the other version: this was replaced by T=torch.cumsum(T,dim = -1)
            T = 1 - torch.cumsum(T,dim = -1)
        
        shift = torch.matmul((T-e_0).unsqueeze(-2), theta.unsqueeze(-1)).squeeze(-1)
        y_orthogonal = m_0 + shift
        return y_orthogonal
    def loss(self, y, treatment, m_0, e_0, theta,active_entries):
        ##Process data to compute the losses
        loss_reg = self.loss_m_0(y,m_0,active_entries)
        loss_e_0 = self.loss_e_0(treatment,e_0,active_entries)
        loss_orthogonal = self.loss_orthogonal(y,treatment,m_0,e_0,theta,active_entries)
        return loss_reg,loss_e_0,loss_orthogonal



    def loss_m_0(self,y,m_0,active_entries):
        """Loss function for the m_0 model

        Args:
            y (torch.tensor): tensor of size (B,L,tgt_size), outcome tensor
            m_0 (torch.tensor): tensor of size(B,L,tgt_size), m_0 tensor
            active_entries (torch.tensor): tensor of shape (B,L,1), indicator of the relevent timesteps

        Returns:
            _type_: _description_
        """
        mse = F.mse_loss(y,m_0,reduction="none")
        loss_regression = (mse*active_entries).sum()/active_entries.sum()
        return loss_regression
    def loss_e_0(self,treatment,e_0,active_entries):
        """loss function for the e_0 model

        Args:
            treatment (torch.tensor): tensor of shape (B,L,1)
            e_0 (torch.tensor): tensor of shape (B,L,2**treatment_max)
            active_entries (torch.tensor): tensor of shape (B,L,1), indicator of the relevent timesteps

        Returns:
            torch.tensor: loss for e_0 model
        """
        if self.is_cdf:
            T_reduced = treatment[:,:,0].long()
            T_reduced = F.one_hot(T_reduced, 2**self.treatment_max).float()
            #T_reduced = torch.cumsum(T_reduced,dim = -1)
            T_reduced = 1 - torch.cumsum(T_reduced,dim = -1)

            # in best version: next 2 lines were replace by single line below
            # loss = self.loss_classification(e_0.flatten(), T_reduced.flatten(), reduction = "none")*active_entries.flatten().sum()/active_entries.sum()
            loss = torch.sum(self.loss_classification(e_0, T_reduced, reduction = "none"),dim=-1)
            loss = (loss.flatten()*active_entries.flatten()).sum()/active_entries.sum()
            return loss
        else:
            return (self.loss_classification(e_0.reshape(-1,2**self.treatment_max), treatment.flatten().long(), reduction = "none")*active_entries.flatten()).sum()/active_entries.sum()
    def loss_orthogonal(self,y, treatment, m_0, e_0, theta,active_entries):
        """_summary_

        Args:
            y (torch.tensor): tensor of size (B,L,tgt_size), outcome tensor
            treatment (torch.tensor): tensor of shape (B,L,1)
            m_0 (torch.tensor): tensor of size(B,L,tgt_size), estimator of treatment
            e_0 (torch.tensor): tensor of shape (B,L,2**treatment_max), estimator of treatment
            theta (torch.tensor): tensor of shape(B,L,2**treatment_max) , ATE
            active_entries (torch.tensor): tensor of shape (B,L,1), indicator of the relevent timesteps

        Returns:
            torch.tensor: R-Loss
        """
        
        
        T_reduced = treatment[:,:,0].long()
        T_reduced = F.one_hot(T_reduced, 2**self.treatment_max).float()
        if self.is_cdf:
            T_reduced = 1 - torch.cumsum(T_reduced,dim = -1)
            #T_reduced = torch.cumsum(T_reduced,dim = -1)
        shift = torch.matmul((T_reduced-e_0).unsqueeze(-2), theta.unsqueeze(-1)).squeeze(-1)
        y_orthogonal = m_0 + shift
        mse = F.mse_loss(y,y_orthogonal,reduction="none")
        loss = (mse*active_entries).sum()/active_entries.sum()
        return loss

    def training_step(self, batch, batch_idx):
        
        vitals = batch['vitals'].float()
        static_features = batch['static_features'].float()
        treatments = batch['current_treatments'].float()
        Y = batch["outputs"].float()
        active_entries = torch.zeros_like(Y)
        li_insample_y = batch["outputs"].clone().float()

        ## Number of hours passed in emergency
        position = torch.arange(batch['vitals'].shape[1])
        position = position.repeat(batch['vitals'].shape[0],1,1)
        position = torch.permute(position,(0,2,1)).to(batch['vitals'].device)
        
        ## Sampling a different tau for each patient of the batch
        taus = batch["future_past_split"].int()
        for i,tau in enumerate(taus):
            li_insample_y[i,tau:] = 0
            active_entries[i,tau:tau+self.proj_len] = 1
            vitals[i,tau:] = 0
        index = taus
        ## Setting the treatment tensor and removing the treatments from the temporal features
        temporal = torch.concat([vitals,position,treatments],dim=-1)
        treatment = torch.zeros_like(Y)
        for k in range(self.treatment_max):
            treatment += temporal[:,:,-1-k].clone().unsqueeze(-1)*(2**k)
            for i,tau in enumerate(index):
                temporal[i,tau:,-1-k] = -1
        ## Encapsulating inputs
        windows = {}
        windows["insample_y"] = li_insample_y
        windows["multivariate_exog"] = temporal
        windows["stat_exog"] = static_features
        
        m, e, theta = self.forward(windows,index)
        loss_reg,loss_e_0,loss_orthogonal= self.loss(Y, treatment,
                                                      m,e,theta,active_entries 
                                                      )
        self.log("train_loss_reg_m_0", loss_reg, on_epoch=True,sync_dist=True)
        self.log("train_loss_e_0", loss_e_0, on_epoch=True,sync_dist=True)
        self.log("train_orthogonal_loss", loss_orthogonal, on_epoch=True,sync_dist=True)
        # If we train m_0 and e_0 we compute the loss by taking their losses
        if self.training_m_e:
            loss = loss_reg+loss_e_0
        else:
            loss = loss_orthogonal
        self.log("train_loss", loss, on_epoch=True,prog_bar=True,sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        
        vitals = batch['vitals'].float()
        static_features = batch['static_features'].float()
        treatments = batch['current_treatments'].float()
        Y = batch["outputs"].float()
        active_entries = torch.zeros_like(Y)
        li_insample_y = batch["outputs"].clone().float()

        ## Number of hours passed in emergency
        position = torch.arange(batch['vitals'].shape[1])
        position = position.repeat(batch['vitals'].shape[0],1,1)
        position = torch.permute(position,(0,2,1)).to(batch['vitals'].device)
        ## We extract the tau from the batch for the validation step
        for i,tau in enumerate(batch["future_past_split"].int()):
            li_insample_y[i,tau:] = 0
            active_entries[i,tau:tau+self.proj_len] = 1
            vitals[i,tau:] = 0
        index = batch["future_past_split"].int()
        
        ## Setting the treatment tensor and removing the treatments from the temporal features
        temporal = torch.concat([vitals,position,treatments],dim=-1)
        treatment = torch.zeros_like(Y)
        for k in range(self.treatment_max):
            treatment += temporal[:,:,-1-k].clone().unsqueeze(-1)*(2**k)
            for i,tau in enumerate(index):
                temporal[i,tau:,-1-k] = -1
        # Encapsulating inputs
        windows = {}
        windows["insample_y"] = li_insample_y
        windows["multivariate_exog"] = temporal
        windows["stat_exog"] = static_features
        
        
        m, e, theta = self.forward(windows,index)

        loss_reg,loss_e_0,loss_orthogonal= self.loss(Y, treatment,
                                                      m,e,theta,active_entries 
                                                      )
        self.log("val_loss_reg_m_0", loss_reg, on_epoch=True,sync_dist=True)
        self.log("val_loss_e_0", loss_e_0, on_epoch=True,sync_dist=True)
        self.log("val_orthogonal_loss", loss_orthogonal, on_epoch=True,sync_dist=True)
        # If we train m_0 and e_0 we compute the loss by taking their losses
        if self.training_m_e:
            loss = loss_reg+loss_e_0
        else:
            loss = loss_orthogonal
        self.log("val_loss", loss, on_epoch=True,prog_bar=True,sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Required by torch lightning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate,weight_decay=self.weight_decay)
        return self.optimizer










