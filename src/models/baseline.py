from src.models.common import *
from src.models.layers_alt import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Callable



class baseline(Transformer_core):
    def __init__(self, h=91,
                proj_len = 5,
                 tgt_size: int = 1,
                 n_features_stat=28, 
                 multi_input_size=4, 
                 hidden_size: int = 64, 
                 n_head: int = 4,
                 learning_rate:float = 0.001,
                 attn_dropout: float = 0, 
                 dropout: float = 0.1, 
                 n_categorical_stat:int = None, 
                 embedding_size_stat:list = None, 
                 n_categorical:int = None, 
                 embedding_size:list = None,
                 last_nn = [256,256],
                 serie_size:int = 1,
                 n_att_layers:int=4,
                 n_static:int=2,
                 padding:int=128,
                 n_block:int=2,
                 kernel_size:int = 5

                 ):
        """Baseline computing the demand with a cumulative loss

        Args:
            h (int, optional): length of the sequence. Defaults to 91.
            tgt_size (int, optional): Number of channel for the output. Defaults to 1.
            n_features_stat (int, optional): Number of static features. Defaults to 28.
            multi_input_size (int, optional): Number of temporal features, without the demand. Defaults to 4.
            hidden_size (int, optional): Hidden size of the network. Defaults to 64.
            n_head (int, optional): NUmber of heads for the self-attention. Defaults to 4.
            learning_rate (float, optional): _description_. Defaults to 0.001.
            attn_dropout (float, optional): Dropout in the self-attention layers. Defaults to 0.
            dropout (float, optional): _description_. Defaults to 0.1.
            n_categorical_stat (int, optional): Number of categorical data in the static mete-data. Defaults to None.
            embedding_size_stat (list, optional): Sizes of the categorical static data. Defaults to None.
            n_categorical (int, optional): NUmber of categorical data for the temporal data. Defaults to None.
            embedding_size (list, optional): Size of the categoricial temporal data. Defaults to None.
            last_nn (list, optional): Size of the last NN after TFT. Defaults to [128].
            serie_size (int, optional): Size of the serie data, 1 if we only compute the demand. Defaults to 1.
            n_att_layers (int, optional): Number of attention layers. Defaults to 4.
            n_static (int, optional): Number of bocks for the static encoder. Defaults to 2.
        """


        super().__init__(h,  tgt_size, n_features_stat, multi_input_size+1, hidden_size,
                         n_head, attn_dropout, dropout, learning_rate, n_categorical_stat, embedding_size_stat, n_categorical,
                         embedding_size, serie_size,n_att_layers,n_static,padding,n_block,kernel_size)
        

        self.loss_regression = nn.MSELoss()
        self.proj_len = proj_len
        #Creatin of the final dense network
        self.last_nn = create_sequential(last_nn, self.hidden_size, tgt_size)
        #self.automatic_optimization = False
        self.configure_optimizers()
        self.save_hyperparameters()
    def forward(self, window_batch,indexes):
        Z= self.get_Z(window_batch,indexes)
        y_hat = self.last_nn(Z)
        return y_hat


    def forecast(self,batch):
        with torch.no_grad():
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


            for i,tau in enumerate(batch["future_past_split"].int()):
                li_insample_y[i,tau:] = 0
                active_entries[i,tau:tau+self.proj_len] = 1
                vitals[i,tau:] = 0
            index = batch["future_past_split"].int()
            
            temporal = torch.concat([vitals,position,treatments],dim=-1)
            # Encapsulating inputs
            windows = {}
            windows["insample_y"] = li_insample_y
            windows["multivariate_exog"] = temporal
            windows["stat_exog"] = static_features
            output = self.forward(windows,index.int())
        return output
        
    def loss(self,y,y_hat,active_entries):
        """This function computes the losses for the m_0 MLP
        """
        loss =  F.mse_loss(y_hat, y, reduction="none")
        loss = (active_entries*loss).sum()/active_entries.sum()
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
        #taus = vectorize_sample_t(batch["sequence_lengths"].cpu().numpy(),self.proj_len)
        taus = batch["future_past_split"].int()
        li_tau = []
        for i,tau in enumerate(taus):
            li_insample_y[i,tau:] = 0
            active_entries[i,tau:tau+self.proj_len] = 1
            vitals[i,tau:] = 0
            li_tau.append(torch.tensor(tau).unsqueeze(-1))
        #index = torch.concat(li_tau)
        index = taus

        temporal = torch.concat([vitals,position,treatments],dim=-1)
        # Encapsulating inputs
        windows = {}
        windows["insample_y"] = li_insample_y
        windows["multivariate_exog"] = temporal
        windows["stat_exog"] = static_features
        #Frist update
        
        output = self.forward(windows,index.int())
        loss =  self.loss(output,Y,active_entries)
        self.log("train_loss", loss, on_epoch=True,prog_bar=True,sync_dist=True)
        

        return loss 
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
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


            for i,tau in enumerate(batch["future_past_split"].int()):
                li_insample_y[i,tau:] = 0
                active_entries[i,tau:tau+self.proj_len] = 1
                vitals[i,tau:] = 0
            index = batch["future_past_split"].int()
            
            temporal = torch.concat([vitals,position,treatments],dim=-1)
            # Encapsulating inputs
            windows = {}
            windows["insample_y"] = li_insample_y
            windows["multivariate_exog"] = temporal
            windows["stat_exog"] = static_features
            output = self.forward(windows,index.int())
            loss =  self.loss(output,Y,active_entries)
            self.log(f"val_loss",loss,on_epoch=True,prog_bar=True,sync_dist=True)
        
        self.train()

    def test_step(self, batch, batch_idx):


        self.eval()
        with torch.no_grad():
            vitals = batch['vitals'].float()
            static_features = batch['static_features'].float()
            treatments = batch['current_treatments'].float()
            position = torch.arange(batch['vitals'].shape[1])
            position = position.repeat(batch['vitals'].shape[0],1,1)
            position = torch.permute(position,(0,2,1)).to(batch['vitals'].device)
            li_insample_y = batch["outputs"].clone().float()
            active_entries = torch.zeros_like(li_insample_y)
            for i in range(batch['future_past_split'].shape[0]):
                vitals[i, int(batch['future_past_split'][i]):] = 0.0
                li_insample_y[i,int(batch['future_past_split'][i]):] = 0
                active_entries[i,int(batch['future_past_split'][i]):int(batch['future_past_split'][i])+self.proj_len] = 1
            static_features = batch['static_features'].float()
            treatments = batch['current_treatments'].float()
            index = batch["future_past_split"]
            temporal = torch.concat([vitals,position,treatments],dim=-1)
            # Encapsulating inputs
            windows = {}
            windows["insample_y"] = li_insample_y
            windows["multivariate_exog"] = temporal
            windows["stat_exog"] = static_features
            output = self.forward(windows,index.int())
            loss =  F.mse_loss(output, batch["outputs"].float(), reduce=False)
            loss = (active_entries*loss).sum()/active_entries.sum()
            self.log(f"RMSE",loss.sqrt(),on_epoch=True,prog_bar=True,sync_dist=True)
        
        self.train()
    def configure_optimizers(self):
        # Required by torch lightning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer


