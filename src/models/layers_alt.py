import torch
from torch.nn import LayerNorm
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor
import pytorch_lightning as pl
import math
from abc import ABC, abstractmethod

from encodec.modules.conv import SConvTranspose1d,SConv1d
from torch.optim.lr_scheduler import StepLR




def default(val, d):
    if not (val is None):
        return val
    return d
class MaybeLayerNorm(nn.Module):
    def __init__(self, output_size, hidden_size, eps):
        super().__init__()
        if output_size and output_size == 1:
            self.ln = nn.Identity()
        else:
            self.ln = LayerNorm(output_size if output_size else hidden_size, eps=eps)

    def forward(self, x):
        return self.ln(x)

class GLU(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.lin = nn.Linear(hidden_size, output_size * 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = F.glu(x)
        return x

class GRN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size=None,
        context_hidden_size=None,
        dropout=0,
    ):
        super().__init__()

        self.layer_norm = MaybeLayerNorm(output_size, hidden_size, eps=1e-3)
        self.lin_a = nn.Linear(input_size, hidden_size)
        if context_hidden_size is not None:
            self.lin_c = nn.Linear(context_hidden_size, hidden_size, bias=False)
        self.lin_i = nn.Linear(hidden_size, hidden_size)
        self.glu = GLU(hidden_size, output_size if output_size else hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_size, output_size) if output_size else None

    def forward(self, a: Tensor, c: Optional[Tensor] = None):
        x = self.lin_a(a)
        if c is not None:
            x = x + self.lin_c(c).unsqueeze(1)
        x = F.elu(x)
        x = self.lin_i(x)
        x = self.dropout(x)
        x = self.glu(x)
        y = a if not self.out_proj else self.out_proj(a)
        x = x + y
        x = self.layer_norm(x)
        return x
   
class TFTEmbedding(nn.Module):
    def __init__(
        self, hidden_size, stat_input_size, multi_input_size, tgt_size,
        n_categorical_stat, embedding_size_stat, n_categorical, embedding_size
    ):
        super().__init__()
        # There are 4 types of input:
        # 1. Static continuous
        # 2. Temporal known a priori continuous
        # 3. Temporal observed continuous
        # 4. Temporal observed targets (time series obseved so far)

        self.hidden_size = hidden_size

        self.stat_input_size = stat_input_size
        self.tgt_size = tgt_size
        self.multivariate_size = multi_input_size
        #Design of the Embeddings for the static metadata
        self.n_categorical_stat = n_categorical_stat
        self.embedding_size_stat = embedding_size_stat
        self.embedding_list_stat = nn.ModuleList([nn.Embedding(self.embedding_size_stat[i], hidden_size) for i in range(self.n_categorical_stat)])
        
        #Design of the Embeddings for the temporal data
        self.n_categorical = n_categorical
        self.embedding_size = embedding_size
        if (n_categorical != 0):
            self.embedding_list_future = nn.ModuleList([nn.Embedding(self.embedding_size[i], hidden_size) for i in range(self.n_categorical)])
        for attr, size in [
            ("stat_exog_embedding", stat_input_size),
            ("multi_exog_embedding", multi_input_size),
            ("tgt_embedding", tgt_size),
        ]:
            if size:
                vectors = nn.Parameter(torch.Tensor(size, hidden_size))
                bias = nn.Parameter(torch.zeros(size, hidden_size))
                torch.nn.init.xavier_normal_(vectors)
                setattr(self, attr + "_vectors", vectors)
                setattr(self, attr + "_bias", bias)
            else:
                setattr(self, attr + "_vectors", None)
                setattr(self, attr + "_bias", None)

    def _apply_embedding(
        self,
        cont: Optional[Tensor],
        cont_emb: Tensor,
        cont_bias: Tensor,
        is_stat_exog = False,
        is_multivariate=False
    ):

        #Dimension augmentation for static data
        if (cont is not None and is_stat_exog):
            #Continuous process
            continuous = cont[:,self.n_categorical_stat:]
            cont_emb_continuous = cont_emb[self.n_categorical_stat:]
            cont_bias_continuous = cont_bias[self.n_categorical_stat:]
            continuous_transformed = torch.mul(continuous.unsqueeze(-1), cont_emb_continuous) + cont_bias_continuous
            
            #Categorical process
            categorical = cont[:,:self.n_categorical_stat]
            embedding_representation = []
            if self.n_categorical_stat>0:
                for i in range(self.n_categorical_stat):
                    embedding_representation.append(self.embedding_list_stat[i](categorical[:,i]))
                    embedding = F.gelu(torch.stack(embedding_representation, dim=1))
                    embedding_all = torch.cat([embedding, continuous_transformed], dim = 1)
            else:
                embedding_all = continuous_transformed
            return embedding_all.float()

        #Dimension augmentation for known future data
        elif (cont is not None and is_multivariate):
            #Continuous process
            continuous = cont[:,:,self.n_categorical:]
            cont_emb_continuous = cont_emb[self.n_categorical]
            cont_bias_continuous = cont_bias[self.n_categorical:]
            continuous_transformed = torch.mul(continuous.unsqueeze(-1), cont_emb_continuous) + cont_bias_continuous
            #Categorical process
            categorical = cont[:,:,:self.n_categorical].int()
            embedding_representation = []
            if self.n_categorical>0:
                for i in range(self.n_categorical):
                    embedding_representation.append(self.embedding_list_future[i](categorical[:,:,i]))
                    embedding = F.gelu(torch.stack(embedding_representation, dim=1))
                    embedding = embedding.permute((0,2,1,3))
                    embedding_all = torch.cat([embedding, continuous_transformed], dim = 2)
            else:
                embedding_all = continuous_transformed
            return embedding_all.float()
        return None

    def forward(self, target_inp, stat_exog=None, multi_exog=None, ):
        # temporal/static categorical/continuous known/observed input
        # tries to get input, if fails returns None

        # Static inputs are expected to be equal for all timesteps
        # For memory efficiency there is no assert statement
        stat_exog = stat_exog[:, :] if stat_exog is not None else None
        
        s_inp = self._apply_embedding(
            cont=stat_exog,
            cont_emb=self.stat_exog_embedding_vectors,
            cont_bias=self.stat_exog_embedding_bias,
            is_stat_exog=True
        )
        
        k_inp = self._apply_embedding(
            cont=multi_exog,
            cont_emb=self.multi_exog_embedding_vectors,
            cont_bias=self.multi_exog_embedding_bias,
            is_multivariate=True
        )

        # Temporal observed targets
        # t_observed_tgt = torch.einsum('btf,fh->btfh',
        #                               target_inp, self.tgt_embedding_vectors)
        target_inp = torch.matmul(
            target_inp.unsqueeze(3).unsqueeze(4),
            self.tgt_embedding_vectors.unsqueeze(1),
        ).squeeze(3)
        target_inp = target_inp + self.tgt_embedding_bias

        return s_inp, k_inp, target_inp



class VariableSelectionNetwork(nn.Module):
    def __init__(self,hidden_size,num_inputs,dropout, n_layers=1):
        super().__init__()
        seq = []
        seq.append(nn.Linear(num_inputs*hidden_size,4*hidden_size))
        seq.append(nn.GELU())
        for _ in range(n_layers):
            #seq.append(nn.Linear(num_inputs*hidden_size,num_inputs*hidden_size))
            seq.append(nn.Linear(4*hidden_size,4*hidden_size))
            seq.append(nn.Dropout(dropout))
            seq.append(nn.GELU())
        seq.append(nn.Linear(4*hidden_size,hidden_size))
        self.seq = nn.Sequential(*seq)
        self.lin_c = nn.Linear(hidden_size,hidden_size)
    def forward(self,x,context = None):
        x = x.flatten(start_dim = -2)
        x = self.seq(x)
        if context is not None:
            #x = x + context.unsqueeze(1)
            x = torch.cat([self.lin_c(context).unsqueeze(1),x],dim=-2)
        return x



class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, n_head, hidden_size, example_length, attn_dropout, dropout):
        super().__init__()
        self.n_head = n_head
        assert hidden_size % n_head == 0
        self.d_head = hidden_size // n_head
        self.hidden_size = hidden_size
        self.qkv_linears = nn.Linear(
            hidden_size, (2 * self.n_head +1) * self.d_head, bias=False
        )
        self.q = nn.Linear(hidden_size,self.n_head*self.d_head)
        self.k = nn.Linear(hidden_size,self.n_head*self.d_head)
        self.v = nn.Linear(hidden_size,self.d_head)
        self.out_proj = nn.Linear(self.d_head, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5
        self.register_buffer(
            "_mask",
            torch.triu(
                torch.full((example_length, example_length), float("-inf")), 1
            ).unsqueeze(0),
        )

    def forward(
        self, x: Tensor, mask_future_timesteps: bool = True,return_weights:bool=False,static_features=None
    ) -> Tuple[Tensor, Tensor]:
        # [Batch,Time,MultiHead,AttDim] := [N,T,M,AD]
        #Computation of the queries, keys and values
        context = default(static_features,x)
        bs, t, h_size = x.shape
        q = self.q(x)
        k = self.k(context)
        v = self.v(context)

        q = q.view(bs, t, self.n_head, self.d_head)
        k = k.view(bs, t, self.n_head, self.d_head)
        v = v.view(bs, t, self.d_head)

        # [N,T1,M,Ad] x [N,T2,M,Ad] -> [N,M,T1,T2]
        # attn_score = torch.einsum('bind,bjnd->bnij', q, k)
        #Computation of the context vectores
        attn_score = torch.matmul(q.permute((0, 2, 1, 3)), k.permute((0, 2, 3, 1)))
        attn_score.mul_(self.scale)

        
        #Masking the future
        if mask_future_timesteps:
            attn_score = attn_score + self._mask
        attn_prob = F.softmax(attn_score, dim=3)
        attn_prob = self.attn_dropout(attn_prob)



        attn_vec = torch.matmul(attn_prob, v.unsqueeze(1))
        m_attn_vec = torch.mean(attn_vec, dim=1)
        out = self.out_proj(m_attn_vec)
        out = self.out_dropout(out)
        if return_weights==True:
            return out, attn_vec,attn_prob
        return out, attn_vec

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch,dropout=0,kernel_size = 7 ):
        super().__init__()
        self.cnn1 = SConv1d(in_ch, out_ch,kernel_size=kernel_size,causal=True,stride=1,pad_mode="constant")
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x):
        y = F.gelu(self.cnn1(x.transpose(1,2))).transpose(1,2)
        y = self.norm(y)

        y = self.dropout(y)
        return y[:, ::2, :]##Taking one token out of two

class Resblock(nn.Module):
    def __init__(self,hidden_size,dropout=0.1,kernel_size = 3):
        super().__init__()
        self.cnn1 = SConv1d(hidden_size,hidden_size,kernel_size=kernel_size,causal=True,stride=1,pad_mode="constant")
        self.dropout = nn.Dropout1d(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.cnn2 = SConv1d(hidden_size,hidden_size,kernel_size=kernel_size,causal=True,stride=1,pad_mode="constant")
    
    def forward(self,x):
        y = x.transpose(1,2)
        y = F.gelu(self.cnn1(y))
        y = F.gelu(self.cnn2(y))
        y = self.dropout(self.norm(y.transpose(1,2)))
        y = x + y
        return y





class encoder_block(nn.Module):
    def __init__(self, n_head,in_ch, out_ch, origin_size, dropout=0,attn_dropout=0.1,n_att_layers:int = 1,padding:int=128,kernel_size:int = 7) -> None:
        super().__init__()
        self.context_adanorm = nn.Sequential(
                                           nn.Linear(origin_size, in_ch*2),
                                           nn.GELU())
        self.down = Downsample(in_ch, out_ch,dropout,kernel_size)
        self.norm = nn.LayerNorm(in_ch)
        self.attention = AttentionBlock(n_head,in_ch,padding,attn_dropout,dropout,origin_size,n_att_layers)
        self.resblock1 = Resblock(in_ch,dropout,kernel_size)
        self.resblock2 = Resblock(in_ch,dropout,kernel_size)
    
    def forward(self,x,static_features=None):
        x = self.resblock1(x)
        x = self.attention(x,static_features)
        if not (static_features is None):
            N,_,h = x.shape            
            weights_stat = self.context_adanorm(static_features).reshape(N,1,h,2)
            x = weights_stat[:,:,:,0]*self.norm(x) + weights_stat[:,:,:,1]
        x = self.resblock2(x)
        x = self.down(x)
        return x

class encoder_CNN(nn.Module):
    def __init__(self, hidden_size, num_vars_temporal,n_head, dropout=0.1,padding=92,h=91,serie_size = 21,n_block = 2,n_att_layers=1,attn_dropout=0.05,
                 kernel_size = 7):
        """This model is the encoder of the temporal data.

        Args:
            hidden_size (_type_): _description_
            num_vars_temporal (int): Number of temporal data
            dropout (_type_): _description_
        """
        super().__init__()

        self.serie_size = serie_size
        self.padding = nn.ZeroPad1d((0,padding-h))

        #Features selection for temporal features known everwhere
        self.temporal_vsn = VariableSelectionNetwork(
            hidden_size=hidden_size, num_inputs=num_vars_temporal-self.serie_size, dropout=dropout
        )
        #Features selection for temporal features known until tau
        self.past_vsn = VariableSelectionNetwork(
            hidden_size=hidden_size, num_inputs=num_vars_temporal, dropout=dropout,
        )

        in_ch = hidden_size
        self.li_down = []
        for _ in range(n_block):
            self.li_down.append(
                encoder_block(n_head,in_ch, in_ch+hidden_size,hidden_size,dropout,attn_dropout,n_att_layers,padding,kernel_size)
            )
            in_ch += hidden_size
            padding = padding//2
        self.li_down = nn.ModuleList(self.li_down)

        
        
    def forward(self, temporal_features, cs,tau=None):
        """_summary_

        Args:
            temporal_features (Tensor): Temporal data, shape (batch_size, temporal lenght,n_features_temporal,hidden_size)
            cs (Tensor): Static data, added to the sequence after the convolution, shape (batch_size, hidden_size)
            tau (int, optional): Time step up until we know the time series. Defaults to None.
            openings(tensor, optional): Tensor of size (Batch_size), indicating the day of the sales' opening 

        Returns:
            _type_: _description_
        """
        #We apply a first features selection to the features known for the whole time series

        x = self.temporal_vsn(temporal_features[:,:,:-self.serie_size])
        N,L,h = x.shape
        #We then add the effect of the features known until tau
        if tau is not None:
            past_effect = self.past_vsn(temporal_features)
            for i in range(tau.shape[0]):
                x[i,:tau[i]] = x[i,:tau[i]] + past_effect[i,:tau[i]]


        #We apply the convolutions
        x = self.padding(x.transpose(1,2)).transpose(1,2)
        li_intermediate = [x]
        for i in range(len(self.li_down)):
            x = self.li_down[i](x,cs)
            if i!=len(self.li_down)-1:
                li_intermediate.append(x)
        

        y = x
        li_intermediate.append(y)
        return li_intermediate



class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch,dropout=0.1,kernel_size = 7):
        super().__init__()
        self.cnn = SConvTranspose1d(in_ch,out_ch,kernel_size=kernel_size,stride=2,causal=True)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(out_ch)
    def forward(self,x):
        y = x.transpose(1,2)
        y = F.gelu(self.cnn(y))
        y = y.transpose(1,2)
        y = self.dropout(self.norm(y))
        
        return y

class decoder_block(nn.Module):
    def __init__(self, n_head, in_ch, out_ch, origin_size,attn_dropout=0.1, dropout=0.1,n_att_layers:int = 1,padding:int=128,kernel_size:int=7) -> None:
        super().__init__()
        self.up =  Upsample(in_ch, out_ch,dropout,kernel_size)
        self.norm1 = nn.LayerNorm(out_ch)
        self.resblock1 = Resblock(out_ch,dropout,kernel_size)
        self.resblock2 = Resblock(out_ch,dropout,kernel_size)
        self.attention = AttentionBlock(n_head,out_ch,padding,attn_dropout,dropout,n_layers =n_att_layers)
        self.norm2 = nn.LayerNorm(out_ch)
        self.context_adanorm = nn.Sequential(
                                           nn.Linear(origin_size, out_ch*2),
                                           nn.GELU())
    def forward(self,x,skip_connection, static_features=None):
        x = self.up(x)
        x = self.resblock1(x)
        if not (static_features is None):
            N,_,h = x.shape            
            weights_stat = self.context_adanorm(static_features).reshape(N,1,h,2)
            x = weights_stat[:,:,:,0]*self.norm1(x) + weights_stat[:,:,:,1]
        x = self.norm2(x+skip_connection)
        x = self.resblock2(x)
        x = self.attention(x)
        return x

class decoder_CNN(nn.Module):
    def __init__(
        self, n_head, hidden_size, h, attn_dropout=0.1, dropout=0.1,n_att_layers:int = 1,padding:int=128,n_block = 2,kernel_size:int = 7

    ):
        """This class gathers the transformer bottleneck of the model and the CNN decoder

        Args:
            n_head (_type_): _description_
            hidden_size (_type_): _description_
            example_length (_type_): _description_
            attn_dropout (_type_): _description_
            dropout (_type_): _description_
            n_att_layers (int, optional): _description_. Defaults to 4.
        """
        super().__init__()
        self.h = h
        
        #Bottleneck attention
        self.attention = AttentionBlock(n_head=n_head,
            hidden_size=hidden_size*(n_block+1),
            example_length=padding//(2**n_block),
            attn_dropout=attn_dropout,
            dropout=dropout,
            n_layers=n_att_layers,
            static_size = hidden_size
        )




        #Upsampling blocks
        in_ch = hidden_size*(n_block+1)
        padding = padding//(2**n_block)*2
        li_up = []
        for _ in range(n_block):
            li_up.append(decoder_block(n_head, in_ch, in_ch-hidden_size, hidden_size, attn_dropout=attn_dropout, dropout=dropout,n_att_layers=n_att_layers,padding=padding,kernel_size=kernel_size))
            in_ch -= hidden_size
            padding = padding * 2
        self.li_up = nn.ModuleList(li_up)
        

        
        
        

    def forward(self,temporal_features,static_features):
        """_summary_

        Args:
            temporal_features (list): List of tensor for the skip connection
            tau (int, optional): Index up until which we know the time series. Defaults to 0.
        """
   
        # Temporal self attention
        enriched = temporal_features[-1]
        x = self.attention(enriched)
        for i in range(1,len(self.li_up)+1):
            x = self.li_up[i-1](x,temporal_features[-i-1],static_features)

        #We only keep the h first tokens because we did padding in the encoder
        x = x[:,:self.h]
        #x =self.final_cnn(x.transpose(2,1)).transpose(1,2)
        return x

class StaticCovariateEncoder(nn.Module):
    def __init__(self, hidden_size, num_static_vars, dropout, n_layers):
        """This class is the encoder of the static meta data

        Args:
            hidden_size (int): hidden size of the mode
            num_static_vars (int): Number of static features in the model
            dropout (_type_): _description_
            n_layers (_type_): _description_
        """
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            hidden_size=hidden_size, num_inputs=num_static_vars, dropout=dropout,n_layers=n_layers
        )
        self.context_grns = GRN(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout)
        self.drop_rate = dropout
    def forward(self, x: Tensor):
        variable_ctx = self.vsn(x)
        context_static = self.context_grns(variable_ctx)
        #If we are training, we randomly set all the static data to zero to prevent the network from focusing to much on the static data
        #if self.training:
        #    context_static = dropout_instance(context_static,self.drop_rate/2)
        return context_static


    
class BasicAttentionBlock(nn.Module):
    def __init__(self,n_head, hidden_size, example_length, attn_dropout, dropout,static_size=None ) -> None:
        super().__init__()
        if static_size==None:
            static_size = hidden_size
        self.attention1 = InterpretableMultiHeadAttention(
                                    n_head=n_head,
                                    hidden_size=hidden_size,
                                    example_length=example_length,
                                    attn_dropout=attn_dropout,
                                    dropout=dropout)
        self.static_encoder = nn.Linear(static_size,hidden_size)
        self.attention2 = InterpretableMultiHeadAttention(
                                    n_head=n_head,
                                    hidden_size=hidden_size,
                                    example_length=example_length,
                                    attn_dropout=attn_dropout,
                                    dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x, static_features = None):
        #attention standard
        N,L,h = x.shape
        x = x + self.attention1(self.norm1(x),mask_future_timesteps=True)[0]
        #Conditional attention
        if not (static_features is None):
          N,H = static_features.shape
          static_features = static_features.unsqueeze(1).expand(N,L,H)
          x = x + self.attention2(self.norm2(x),mask_future_timesteps=True,
                                  static_features=F.relu(self.static_encoder(static_features)))[0]
        return x


 


class AttentionBlock(nn.Module):

    def __init__(self, n_head, hidden_size, example_length, attn_dropout, dropout, static_size=None, n_layers = 4):
        """This class is one attention block

        Args:
            n_head (int): NUmber of heads of attentions
            hidden_size (int): Hidden size of the model
            example_length (int): Length of the sequence
            attn_dropout (float): Dropout in the layer attention
            dropout (float): Dropout in other layers
            n_layers (int, optional): Number of attention layers. Defaults to 4.
        """
        super(AttentionBlock,self).__init__()
        li = []
        for _ in range(n_layers):
            li.append(BasicAttentionBlock(
                                    n_head=n_head,
                                    hidden_size=hidden_size,
                                    example_length=example_length,
                                    attn_dropout=attn_dropout,
                                    dropout=dropout,
                                    static_size=static_size)
            )
            li.append(LayerNorm(normalized_shape=hidden_size))
        self.li_attention = nn.ModuleList(li)
    
    def forward(self, x, static_features=None):
        for i in range(len(self.li_attention)//2):
            out = self.li_attention[2*i](x,static_features)
            x = x + self.li_attention[2*i+1](out) 
        return x


class Transformer_core(pl.LightningModule, ABC):
    def __init__(self,
                 h=91,
                tgt_size: int = 95+1,
                n_features_stat=28,
                multi_input_size = 4,
                hidden_size: int = 128,
                n_head: int = 4,
                attn_dropout: float = 0.1,
                dropout: float = 0.1,
                learning_rate: float = 0.001,
                n_categorical_stat:int = None, 
                embedding_size_stat:list = None, 
                n_categorical:int = None, 
                embedding_size:list = None,
                serie_size:int =1,
                n_att_layers:int = 4,
                n_stat_layers:int= 2,
                padding:int = 128,
                n_block:int = 2,
                kernel_size:int = 7

    ):
        """This class is an abstract class. It contains the key components of a TFT. It is also a pytorch lightning module, this 
        framework of pytorch used to quickly train the models. This class is used as a parent of every model we work on.

        Args:
            h (int, optional): Horizon of the prediction, how many days we want to predict at most. Defaults to 91.
            input_size (int, optional): Size of the window of the past, at best we used the first 90 days to predict the last 0. Defaults to 90.
            tgt_size (int, optional): Number of head at the end of the network. Defaults to 1.
            n_features_stat (int, optional): Number of features in the static metadata. Defaults to 28.
            futur_exog_size (int, optional): Number of features in the known data. Defaults to None.
            hidden_size (int, optional): Size of the hidden state of the network. Defaults to 256.
            n_head (int, optional): Number of head for the attention mechanism. Defaults to 4.
            attn_dropout (float, optional): Dropout rate for the attention mechanism. Defaults to 0.0.
            dropout (float, optional): _description_. Defaults to 0.1.
            learning_rate (float, optional): initial lr. Defaults to 1e-3.
            n_categorical_stat (int, optional): Number of categorical features in the metadata. Defaults to None.
            embedding_size_stat (list, optional): List of the maximum values for the categorical features in the metadata. Defaults to None.
            n_categorical_future (int, optional): Number of categorical features in the future data. Defaults to None.
            embedding_size_future (list, optional): List of the maximum values for the categorical features in the future data. Defaults to None.
            serie_size (int, optional): Dimension of the time series. Defaults to 1.
        """
        super(Transformer_core, self).__init__()
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.stat_size = n_features_stat
        self.serie_size = serie_size
        self.h = h
        self.tgt_size = tgt_size
        self.multi_input_size = multi_input_size
        num_vars_temporal = self.multi_input_size + self.serie_size 
        self.n_categorical_stat = n_categorical_stat
        self.embedding_size_stat = embedding_size_stat
        self.n_categorical = n_categorical
        self.embedding_size = embedding_size
        self.embedding = TFTEmbedding(
            hidden_size=self.hidden_size,
            stat_input_size=self.stat_size,
            multi_input_size=self.multi_input_size,
            tgt_size=self.serie_size,
            n_categorical_stat=  self.n_categorical_stat,
            embedding_size_stat= self.embedding_size_stat,
            n_categorical=  self.n_categorical,
            embedding_size= self.embedding_size,

        )
        self.static_encoder = StaticCovariateEncoder(
            hidden_size=self.hidden_size, num_static_vars=self.stat_size, dropout=dropout,n_layers=n_stat_layers
        )


        self.temporal_encoder = encoder_CNN(
            hidden_size=self.hidden_size,
            num_vars_temporal = num_vars_temporal,
            n_head=n_head,
            dropout=dropout,
            padding=padding, 
            h=self.h,
            serie_size=self.serie_size,
            n_block=n_block,
            n_att_layers = n_att_layers,
            attn_dropout=attn_dropout,
            kernel_size=kernel_size
        )

        # ------------------------------ Decoders -----------------------------#
        self.temporal_fusion_decoder = decoder_CNN(
            n_head=n_head,
            hidden_size=self.hidden_size ,
            h=self.h,
            attn_dropout=attn_dropout,
            dropout=dropout,
            n_att_layers=n_att_layers,
            padding = padding,
            n_block=n_block,
            kernel_size=kernel_size
        )




    
    def get_Z(self, window_batch,tau):
        """Function returning the latent variable of the batch. It is called by children classes and also by the RL environment.

        Args:
            window_batch (dict): contains all the data of the batch

        Returns:
            tensor: latent variable, output of the TFT
        """
        y_insample =  window_batch["insample_y"]
        multivariate =  window_batch["multivariate_exog"]
        stat_exog = window_batch["stat_exog"]
        if tau is None:
            tau = torch.argmin(y_insample,dim=1)
        # Inputs embeddings
        s_inp, k_inp, t_observed_tgt = self.embedding(
            target_inp=y_insample,
            multi_exog=multivariate,
            stat_exog=stat_exog,
        )

        # -------------------------------- Inputs ------------------------------#
        # Static context

        static_features = self.static_encoder(s_inp)
        temporal_features = torch.cat([k_inp,t_observed_tgt],dim=-2)#We add the time serie to the temporale features
        #temporal_features = torch.cat([multivariate,y_insample],dim=-1)
        
        # ---------------------------- Encode/Decode ---------------------------#
        # CNN
        temporal_features = self.temporal_encoder(
            temporal_features=temporal_features,
            cs=static_features,
            tau=tau
        )

        
        # Self-Attention decoder
        Z = self.temporal_fusion_decoder(
            temporal_features=temporal_features,
            static_features = static_features
        )



        return Z

    
    
    def forward(self, windows_batch):
        pass
    
    def predict(self, shaped_batch):
        li_insample_y,multivariate_exo,li_stat_exo = shaped_batch
        windows = {}
        windows["insample_y"] = li_insample_y
        windows["stat_exog"] = li_stat_exo
        windows["multivariate_exog"] = multivariate_exo
        return self.forward(windows)
    
    def configure_optimizers(self):
        # Required by torch lightning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=20, gamma=0.05)
        return self.optimizer