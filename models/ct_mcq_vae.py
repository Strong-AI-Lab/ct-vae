import torch
import torch_geometric
from models import BaseVAE, MultipleCodebookVectorQuantizer, ResidualLayer
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from .types_ import *
import math
from typing import Union, List



def detach_and_paste(func):
    def inner(_self: nn.Module, x : Tensor, *args, **kwargs):
        # y, *remain = func(_self, x, *args, **kwargs)
        y, *remain = func(_self, x.detach(), *args, **kwargs)
        return x + (y - x).detach(), *remain
    return inner



class PositionalEncoding(nn.Module):
    """
    Inspired by <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0,1)
        x = x + self.pe[:x.size(0)].to(x.device)
        x = x.transpose(0,1)
        return self.dropout(x)



class CausalTransition(nn.Module):

    def __init__(self, 
                 input_dim: int,
                 action_dim: int,
                 latent_dim: int = 800,
                 nb_heads: int = 1,
                 noise: str = "off",
                 c_alpha: float = 0.7,
                 c_beta: float = 0.4,
                 c_gamma: float = 0.9,
                 c_delta: float = 0.4,
                 c_epsilon: float = 0.4,
                 comp_adj_optim: str = "comp",
                 **kwargs) -> None:
        """
        :param noise: (str) Noise in the transition process. 
                        "off": no noise is  added, default. 
                        "exo": noise is added as an exogenous factor affecting all causal variables.
                        "endo": noise is an extra endogenous variable in the causal graph.
        :param c_alpha: (float) Factor leveraging the trend towards identity behaviour when no causal changes.
        :param c_beta: (float) Variational loss.
        :param c_gamma: (float) Factor leveraging the loss of the adjacency matrix coefficients.
        :param c_delta: (float) Factor regularising the size of the causal graph.
        :param c_epsilon: (float) Factor leveraging the confidence that the learned adjacency matrix is not empty
        :param comp_adj_optim: (str) Computation tradeoff for graph discovery optimization. 
                        "comp": All variables are loaded in memory at once, reducing the number of computations needed but needing a higher amount of memory, default. 
                        "mem": One variable is loaded at a time, reducing the memory consumption but needing more computations, increasing the time and the size of the computation graph. 
        """
        super(CausalTransition, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.noise = noise
        self.alpha = c_alpha
        self.beta = c_beta
        self.gamma = c_gamma
        self.delta = c_delta
        self.epsilon = c_epsilon
        self.comp_adj_optim = self._compute_adj__mem_optim if comp_adj_optim == "mem" else self._compute_adj__comp_optim

        self.a_dense = nn.Linear(action_dim, input_dim) # Action formatting for computation along other causal variables

        self.pos_encoding = PositionalEncoding(input_dim) # Positional encodingfor causal variables

        a_discovers = []
        for _ in range(action_dim + 1):
            a_discovers.append(
                    nn.Sequential(
                    nn.Linear(2 * input_dim, latent_dim),
                    nn.LeakyReLU(),
                    nn.Linear(latent_dim, 1),
                    nn.Sigmoid()
                    ))
        
        y_discovers = []
        for _ in range(input_dim + 1):
            y_discovers.append(
                    nn.Sequential(
                    nn.Linear(2 * input_dim, latent_dim),
                    nn.LeakyReLU(),
                    nn.Linear(latent_dim, 1),
                    nn.Sigmoid()
                    ))

        self.graph_discovers = nn.ModuleDict({
                "a" : nn.ModuleList(a_discovers), # Computation of adjacency matrix coefficients in action mode
                "y" : nn.ModuleList(y_discovers) # Computation of adjacency matrix coefficients in causal mode
            })

        self.masks = nn.ModuleDict({
                "a" : nn.Sequential(
                        nn.Linear(action_dim, input_dim),
                        nn.Sigmoid()
                    ), # Action intervention masking, selects adjacency matrix discoverer corresponding to action a
                "y" : nn.Sequential(
                        nn.Linear(input_dim, input_dim),
                        nn.Sigmoid()
                    ) # Causal intervention masking, selects adjacency matrix discoverer corresponding to output y
            })

        self.nb_heads = nb_heads
        self.graph_transitioner = gnn.Sequential('x, edge_index, edge_attr', [
                                    (gnn.GATv2Conv(input_dim, latent_dim // self.nb_heads, edge_dim=1, heads=self.nb_heads), 'x, edge_index, edge_attr -> x'),
                                    nn.ReLU(inplace=True),
                                    (gnn.GATv2Conv(latent_dim, latent_dim // self.nb_heads, edge_dim=1, heads=self.nb_heads), 'x, edge_index, edge_attr -> x'),
                                    nn.ReLU(inplace=True),
                                    (gnn.GATv2Conv(latent_dim, latent_dim // self.nb_heads, edge_dim=1, heads=self.nb_heads), 'x, edge_index, edge_attr -> x'),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(latent_dim, input_dim),
                                    nn.Softmax(dim=-1)
                                ]) # Causal graph inference


    def _compute_mask(self, one_hot_latent, aux, mode = "a"): # TODO: assess if mask needed for y mode since already know if variable affected by intervention
        if mode == "a":
            aux = aux.unsqueeze(1).repeat(1,one_hot_latent.size(1), 1)

        inter_mask = self.masks[mode](aux.to(dtype=torch.float32))
        inter_masked_latent = (one_hot_latent * inter_mask ).sum(dim=-1) # [B x HW]
        logits = torch.log(torch.stack([1 - inter_masked_latent, inter_masked_latent],dim=-1).clamp(min=1e-4)) # [B x HW x 2], clamp to avoid -inf values

        mask = F.gumbel_softmax(logits,tau=1,hard=True)[...,1].unsqueeze(-1) # [B x HW x 1]
        return mask

    def _split_by_inter(self, aux):
        inter_ids = torch.argmax(aux, dim=-1) # [B x HW x A] --> [B x HW] if action mode else [B x HW x D] --> [B x HW]
        inter_dict = {id:torch.where(inter_ids==id) for id in set(inter_ids.view((-1)).tolist())}
        return inter_dict

    def _merge_inter(self, inter_dict, inter_coeffs_dict, shape, device):
        res = torch.zeros(shape).to(device)
        for i in inter_dict.keys():
            res[inter_dict[i]] = inter_coeffs_dict[i]
        return res        

    def _compute_adj__comp_optim(self, latent, aux, mask, mode = "a"): # /!\ big memory needed!
        repeats = latent.size(1)
        nodes_i = latent.repeat(1,1,repeats).view((latent.size(0),-1,latent.size(2))) # [B x HW x D] --> [B x HWHW x D]
        nodes_j = latent.repeat(1,repeats,1) # [B x HW x D] --> [B x HWHW x D]
        
        inp = torch.concat([nodes_i,nodes_j],-1) # [B x HWHW x 2D]

        no_inter_coeffs = self.graph_discovers[mode][0](inp).view((-1,repeats,repeats)) # [B x HW x HW]

        if mode == "a":
            aux = aux.unsqueeze(1).repeat(1,inp.size(1), 1)
            
        inter_dict = self._split_by_inter(aux)
        inter_coeffs_dict = {i: self.graph_discovers[mode][1+i](inp[indices]) for i, indices in inter_dict.items()}
        inter_coeffs = self._merge_inter(inter_dict, inter_coeffs_dict, (inp.size(0), inp.size(1), 1), inp.device).view((-1,repeats,repeats))

        return no_inter_coeffs * (1 - mask) + inter_coeffs * mask
    
    def _compute_adj__mem_optim(self, latent, aux, mask, mode = "a"): # /!\ very big computation graph!
        repeats = latent.size(1)
        
        if mode == "a":
            aux = aux.unsqueeze(1).repeat(1,repeats,1) # [B x A] --> [B x HW x A]

        no_inter_coeffs = torch.zeros((latent.size(0), repeats, repeats)).to(latent.device)  # [B x HW x HW]
        inter_coeffs = torch.zeros((latent.size(0), repeats, repeats)).to(latent.device)  # [B x HW x HW]

        for i in range(repeats):
            nodes_i = latent[:,i,:].unsqueeze(1).repeat(1,repeats,1)
            nodes_j = latent
            inp = torch.concat([nodes_i, nodes_j],-1)
            
            no_inter_coeffs[:,i] = self.graph_discovers[mode][0](inp).view((latent.size(0), repeats))

            inter_dict = self._split_by_inter(aux)
            inter_coeffs_dict = {i: self.graph_discovers[mode][1+i](inp[indices]) for i, indices in inter_dict.items()}
            inter_coeffs[:,i] = self._merge_inter(inter_dict, inter_coeffs_dict, (inp.size(0), inp.size(1), 1), inp.device).view((latent.size(0), repeats))
            
        return no_inter_coeffs * (1 - mask) + inter_coeffs * mask

    def _compute_adj(self, latent, aux, mask, mode = "a"):
        return self.comp_adj_optim(latent, aux, mask, mode)


    def _sample_bernoulli(self, adjacency, differentiable=True):
        if differentiable: # reparametrization trick using Straight-through Gumbel-Softmax
            logits = torch.log(torch.stack([1-adjacency,adjacency],dim=-1).clamp(min=1e-4)) # clamp to avoid -inf values
            return F.gumbel_softmax(logits,tau=1,hard=True)[...,1]
        else:
            return torch.bernoulli(adjacency)
    

    def _compute_y(self, latent, action, adjacency):
        # preprocess
        if self.noise == "exo":
            latent = latent + torch.normal(0,1,latent.shape).to(latent.device) # noise integrated to variables
            padding_h = torch.nn.ConstantPad2d((0,0,0,1),0)
            padding_v = torch.nn.ConstantPad2d((0,1,0,0),1)
            var_supp = action.unsqueeze(1)
        elif self.noise == "endo":
            padding_h = torch.nn.ConstantPad2d((0,0,0,2),0)
            padding_v = torch.nn.ConstantPad2d((0,2,0,0),1) # noise as extra variable
            var_supp = torch.stack([action, torch.normal(0,1,action.shape).to(action.device)],dim=1)
        else:
            padding_h = torch.nn.ConstantPad2d((0,0,0,1),0)
            padding_v = torch.nn.ConstantPad2d((0,1,0,0),1)
            var_supp = action.unsqueeze(1)
            
        nodes = torch.concat([latent,var_supp],1).view((-1,latent.size(-1))) # [(BHW+vs) x D]
        padded_adjacency = padding_h(padding_v(adjacency)) # add missing edges to action

        edge_index, edge_attrs = torch_geometric.utils.dense_to_sparse(padded_adjacency) # format to edge_index [2 x E] and edge_attrs [E] 
        
        # Graph Neural Network computation
        nodes_y = self.graph_transitioner(nodes, edge_index, edge_attr=edge_attrs) # [B(HW+vs) x D]
        
        # postprocess
        var_supp_size = var_supp.size(1)
        action_latent_shape = list(latent.shape) # [B x HW x D]
        action_latent_shape[1] += var_supp_size # [B x (HW+vs) x D]

        return nodes_y.view((action_latent_shape))[:,:-var_supp_size,:] # [B x HW x D] (remove action and noise nodes) 


    def _infer_action(self, adjacency_coeffs, latent, differentiable=True):
        a = F.one_hot(torch.arange(self.action_dim).repeat(latent.size(0))).view((latent.size(0),self.action_dim,self.action_dim)).to(latent.device, dtype=latent.dtype) # [B x A x A]
        a = a.permute(1,0,2) # [A x B x A]
        pos_latent = self.pos_encoding(latent)  # add positional embeddings

        distances = torch.zeros((a.size(1),self.action_dim)).to(a.device) # [B x A]
        for i in range(self.action_dim):
            mask = self._compute_mask(latent, a[i], mode="a")
            adj_i = self._compute_adj(pos_latent, a[i], mask, "a")
            adj_i = adj_i.detach().clone() # prevents backpropagation
            distances[:,i] = F.pairwise_distance(adjacency_coeffs.view((a.size(1), -1)), adj_i.view((a.size(1), -1)))
        probas = 1-torch.softmax(distances,dim=-1)
        
        return probas # [B x A]


    # @detach_and_paste
    def forward(self, latent: Tensor, **kwargs) -> List[Tensor]:
        latent_shape = latent.shape # [B x D x H x W]
        latent = latent.permute(0,2,3,1).view(latent_shape[0], -1, latent_shape[1]) # [B x HW x D]

        mask = torch.zeros(latent.size(0), latent.size(1), 1).to(latent.device)
        pos_latent = self.pos_encoding(latent)  # add positional embeddings
        action = torch.zeros(latent.size(0), self.action_dim).to(latent.device) # [B x A]

        # Compute causal graph
        adjacency_coeffs = self._compute_adj(pos_latent, action, mask, "a")
        causal_graph = self._sample_bernoulli(adjacency_coeffs)

        # Infer y on causal graph with GNN
        action = self.a_dense(action) # [B x A]  --> [B x D]
        latent_y = self._compute_y(pos_latent, action, causal_graph) # [B x HW x D]

        # Compute loss function
        id_matrix = F.one_hot(torch.arange(causal_graph.size(-1)).repeat(causal_graph.size(0),1)).to(latent.device, dtype=causal_graph.dtype)
        ct_reg = self.alpha * (
                    F.cross_entropy(self._compute_y(pos_latent, action, id_matrix).reshape((-1, latent_shape[1])).clamp(min=1e-4).log(), latent.reshape((-1, latent_shape[1])).argmax(dim=-1))
                    + F.mse_loss(causal_graph, id_matrix)
                ) + self.gamma * self.dependencies_MSE_loss(adjacency_coeffs, latent, latent_y, "y") + self.epsilon * self.positive_trial_loss(adjacency_coeffs)
        
        latent_y = latent_y.permute(0,2,1).view(latent_shape) # [B x D x H x W]
        return [latent_y, ct_reg, {"ct_adjacency": adjacency_coeffs.mean(0)}]


    # @detach_and_paste
    def forward_action(self, latent: Tensor, action: Tensor, **kwargs) -> List[Tensor]:
        latent_shape = latent.shape # [B x D x H x W]
        latent = latent.permute(0,2,3,1).view(latent_shape[0], -1, latent_shape[1]) # [B x HW x D]

        mask = self._compute_mask(latent, action, mode="a")
        pos_latent = self.pos_encoding(latent)  # add positional embeddings

        # Compute causal graph
        adjacency_coeffs = self._compute_adj(pos_latent, action, mask, "a")
        causal_graph = self._sample_bernoulli(adjacency_coeffs)

        # Infer y on causal graph with GNN
        action = self.a_dense(action) # [B x A] --> [B x D]
        latent_y = self._compute_y(pos_latent, action, causal_graph) # [B x HW x D]

        # Compute loss function
        ct_reg = self.gamma * self.dependencies_MSE_loss(adjacency_coeffs, latent, latent_y, "y") + self.delta * self.graph_size_loss(causal_graph) + self.epsilon * self.positive_trial_loss(adjacency_coeffs)
        
        latent_y = latent_y.permute(0,2,1).view(latent_shape) # [B x D x H x W]
        return [latent_y, ct_reg, {"ct_mask": mask.view(latent_shape[:1]+latent_shape[2:]).mean(0), "ct_adjacency": adjacency_coeffs.mean(0)}]

    
    # no @detach_and_paste since causal model also acts as decoder for this setting
    def forward_transition(self, latent: Tensor, latent_y: Tensor, **kwargs) -> List[Tensor]:
        latent_shape = latent.shape # [B x D x H x W]
        latent = latent.permute(0,2,3,1).view(latent_shape[0], -1, latent_shape[1]) # [B x HW x D]
        latent_y = latent_y.permute(0,2,3,1).view(latent_shape[0], -1, latent_shape[1]) # [B x HW x D]
        
        mask = self._compute_mask(latent, latent_y, mode="y")
        pos_latent = self.pos_encoding(latent)  # add positional embeddings
        
        # Compute causal graph
        adjacency_coeffs = self._compute_adj(pos_latent, latent_y, mask, "y")

        # Infer a
        action_probas = self._infer_action(adjacency_coeffs, latent)
        action = F.one_hot(torch.argmax(action_probas,dim=1))

        # Compute loss function
        causal_graph = self._sample_bernoulli(adjacency_coeffs)
        ct_reg = self.gamma * self.dependencies_MSE_loss(adjacency_coeffs, latent, action, "a") + self.delta * self.graph_size_loss(causal_graph) + self.epsilon * self.positive_trial_loss(adjacency_coeffs)

        return [action_probas, ct_reg, {"ct_mask": mask.view(latent_shape[:1]+latent_shape[2:]).mean(0), "ct_adjacency": adjacency_coeffs.mean(0)}]


    
    def latent_loss(self, latent, latent_y):
        latent_y = latent_y.detach()
        return self. latent_CrossEntropy_loss(latent, latent_y) + self.beta * self.latent_KL_loss(latent, latent_y) 
    
    def latent_MSE_loss(self, latent, latent_y):
        return F.mse_loss(latent, latent_y)
    
    def latent_CrossEntropy_loss(self, latent, latent_y):
        latent = latent.permute(0, 2, 3, 1).reshape((-1, latent.size(1))) # [B x D x H x W] --> [BHW x D]
        latent = latent.clamp(min=1e-4).log()
        latent_y = latent_y.permute(0, 2, 3, 1).reshape((-1, latent_y.size(1))) # [B x D x H x W] --> [BHW x D]
        latent_y = torch.argmax(latent_y, dim=-1)
        return F.cross_entropy(latent, latent_y)
    
    def latent_KL_loss(self, latent, latent_y):
        return F.kl_div(latent.clamp(min=1e-4).log(), latent_y, reduction="batchmean")


    def dependencies_MSE_loss(self, adjacency_coeffs, one_hot_latent, action, mode = "a"):
        adjacency_coeffs = adjacency_coeffs.detach()
        one_hot_latent = one_hot_latent.detach()
        action = action.detach()
        mask = self._compute_mask(one_hot_latent, action, mode)
        pos_latent = self.pos_encoding(one_hot_latent)

        twin_coeffs = self._compute_adj(pos_latent, action, mask, mode)
        return F.mse_loss(twin_coeffs, adjacency_coeffs)
    
    def dependencies_KL_loss(self, causal_graph, one_hot_latent, action, mode = "a"):
        causal_graph = causal_graph.detach()
        one_hot_latent = one_hot_latent.detach()
        action = action.detach()
        mask = self._compute_mask(one_hot_latent, action, mode)
        pos_latent = self.pos_encoding(one_hot_latent)

        twin_cg = self._sample_bernoulli(self._compute_adj(pos_latent, action, mask, mode))
        return F.kl_div(twin_cg.log(), causal_graph, reduction="batchmean")


    def graph_size_loss(self, causal_graph):
        return torch.linalg.norm(causal_graph)
    
    def positive_trial_loss(self, adjacency_coeffs):
        return torch.linalg.norm((1-adjacency_coeffs).prod(1))




class CTMCQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 action_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 causal_hidden_dim: int = 800,
                 causal_nb_heads: int = 1,
                 beta: float = 0.25,
                 gamma: float = 0.25,
                 img_size: int = 64,
                 codebooks:int = 1,
                 **kwargs) -> None:
        super(CTMCQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.in_channels = in_channels
        self.beta = beta
        self.gamma = gamma
        self.codebooks = codebooks

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
        self.nb_latents = self.img_size // 2**len(hidden_dims)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(in_channels, in_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = MultipleCodebookVectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        codebooks,
                                        self.beta)

        self.ct_layer = CausalTransition(num_embeddings, #embedding_dim//codebooks,
                                        action_dim,
                                        causal_hidden_dim,
                                        causal_nb_heads,
                                        **kwargs)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1],
                                   out_channels=self.in_channels,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Tanh()))

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        return [result]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder(z)
        return result


    def ct_preprocess(self, x: Tensor, latents_shape: list) -> Tensor:
        """
        Formats the tensor for causal transition after computing of inds in quantization: 
        encode input inds with one-hot
        break dimensions into K codebooks and concat blocks in sequence
        :param x: (Tensor) [B x K x H x W] 
        :return: (Tensor) [B x N x (K*H) x W]
        """
        x = F.one_hot(x,num_classes=self.num_embeddings).to(dtype=torch.float32) # [B x K x H x W x N]
        x = x.view((latents_shape[0], self.codebooks*latents_shape[2], latents_shape[3], self.num_embeddings)) # [B x (K*H) x W x N]
        x = x.permute(0,3,1,2) # [B x N x (K*H) x W]
        return x
    
    def ct_postprocess(self, x: Tensor, latents_shape: list):
        """
        Formats the tensor for computation of latents in quantization after causal transition: 
        retrieve one-hot classes with argmax
        rebuild the K codebooks
        :param x: (Tensor) [B x N x (K*H) x W]
        :return: (Tensor) [B x K x H x W]
        """
        x = x.permute(0,2,3,1) # [B x (K*H) x W x N]
        x = x.reshape((latents_shape[0], self.codebooks, latents_shape[2], latents_shape[3], self.num_embeddings)) # [B x K x H x W x N]
        x = torch.argmax(x,dim=-1) # [B x K x H x W]
        return x




    def forward_base(self, input: Tensor, **kwargs) -> List[Tensor]:
        # Encoding
        latents = self.encode(input)[0]

        # Quantization indicator searh
        encoding_inds = self.vq_layer.compute_inds(latents) # [B x K x H x W]

        # Causal Transition
        latents_shape = latents.shape # [B x D x H x W] 
        encodings_one_hot = self.ct_preprocess(encoding_inds, latents_shape) # [B x N x (K*H) x W]
        ct_encodings, ct_reg, *ct_metrics = self.ct_layer(encodings_one_hot) # we may need to apply it either before or after full quantization
        ct_loss = ct_reg + self.ct_layer.latent_loss(ct_encodings, encodings_one_hot)
        ct_encodings = self.ct_postprocess(ct_encodings, latents_shape)

        # Quantization latent retrieval
        quantized_latents, vq_loss = self.vq_layer.compute_latents(latents, ct_encodings)

        # Decoding
        return [self.decode(quantized_latents), input, vq_loss, ct_loss, {**{"mode" : "base"}, **ct_metrics[0]}]

        
    def forward_action(self, input: Tensor, action: Tensor, input_y: Tensor = None, **kwargs) -> List[Tensor]:
        # Encoding
        latents = self.encode(input)[0]

        # Quantization indicator searh
        encoding_inds = self.vq_layer.compute_inds(latents) # [B x K x H x W]
        
        # Causal Transition
        latents_shape = latents.shape
        encodings_one_hot = self.ct_preprocess(encoding_inds, latents_shape) # [B x N x (K*H) x W]
        ct_encodings, ct_reg, *ct_metrics = self.ct_layer.forward_action(encodings_one_hot, action)
        ct_loss = ct_reg + self.ct_layer.latent_loss(ct_encodings, self.ct_preprocess(self.vq_layer.compute_inds(self.encode(input_y)[0]), latents_shape))
        ct_encodings = self.ct_postprocess(ct_encodings, latents_shape)

        # Quantization latent retrieval
        quantized_latents, vq_loss = self.vq_layer.compute_latents(latents, ct_encodings)
        
        # Decoding
        return [self.decode(quantized_latents), input_y, vq_loss, ct_loss, {**{"mode" : "action"}, **ct_metrics[0]}]


    def forward_causal(self, input: Tensor, input_y: Tensor, action: Tensor = None, **kwargs) -> List[Tensor]:
        # Encoding
        latents_x = self.encode(input)[0]
        latents_y = self.encode(input_y)[0]

        # Quantization indicator searh
        encoding_x = self.vq_layer.compute_inds(latents_x) # [B x K x H x W]
        encoding_y = self.vq_layer.compute_inds(latents_y) # [B x K x H x W]  /!\ no backpropagation to encoders
        
        # Causal Transition
        latents_shape = latents_x.shape
        encodings_one_hot_x = self.ct_preprocess(encoding_x, latents_shape) # [B x N x (K*H) x W]
        encodings_one_hot_y = self.ct_preprocess(encoding_y, latents_shape) # [B x N x (K*H) x W]
        recons_action, ct_reg, *ct_metrics = self.ct_layer.forward_transition(encodings_one_hot_x, encodings_one_hot_y)
        ct_acc = (torch.argmax(recons_action, dim=-1)==torch.argmax(action, dim=-1)).float().mean()
        
        # Decoding
        return [recons_action, action, torch.tensor(0.0), ct_reg, {**{"causal_acc": ct_acc, "mode": "causal"}, **ct_metrics[0]}]


    FORWARD_MODES = {
        "base": forward_base,
        "action": forward_action,
        "causal": forward_causal
    }
    def forward(self, input: Tensor, input_y: Tensor = None, action: Tensor = None, mode: Union[str,List[str]] = "base", **kwargs) -> List[Tensor]:
        """
        :param input: (Tensor) Input tensor x to process [N x C x H x W]
        :param input_y: (Tensor) Input tensor y after transition caused by action [N x C x H x W]
        :param action: (Tensor)  Tensor responsible for transition from x to y [N x A]
        :param mode: (str) Mode of application of model.
                "base": encode-decode input
                "action": encode input and decode it after action transition
                "causal": from input and input_y after transition, deduce the actions needed for transition
        """
        if type(mode) is list:
            mode = mode[0]
        if input_y is not None:
            input_y = input_y.to(input.device)
        if action is not None:
            action = action.to(input.device)
        return CTMCQVAE.FORWARD_MODES[mode](self, input=input,input_y=input_y,action=action)


    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]
        ct_loss = args[3]
        metrics = {} if len(args) < 5 else args[4]

        if len(metrics) > 0 and "mode" in metrics and metrics["mode"] == "causal": # forward_causal is a classification task while the others are regressions
            recons_loss = F.cross_entropy(recons.clamp(min=1e-4).log(), torch.argmax(input, dim=-1))
        else:
            recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss + self.gamma * ct_loss
        return {
                **{'loss': loss,
                    'Reconstruction_Loss': recons_loss,
                    'VQ_Loss': vq_loss,
                    'CT_Loss': ct_loss},
                **metrics
                }

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.embedding_dim,
                        self.nb_latents,
                        self.nb_latents) # [B x D x H x W]

        z = z.to(current_device)

        quantized_inputs, _ = self.vq_layer(z)
        samples = self.decode(quantized_inputs)
        return samples

    # def walk(self,
    #            num_steps:int,
    #            num_dims:int,
    #            num_walks:int,
    #            current_device: int, **kwargs) -> Tensor:
    #     """
    #     Walks in the latent space and return the corresponding
    #     images generated at each step. Performs num_walks walks 
    #     with num_steps steps modifying num_dims dimensions and 
    #     generates num_walks x num_steps images.
    #     :param num_steps: (Int) Number of steps in the walk
    #     :param num_dims: (Int) Number of dimensions to walk along (for a single walk)
    #     :param num_walks: (Int) Number of walks to perform
    #     :param current_device: (Int) Device to run the model
    #     :return: (Tensor)
    #     """
    #     z = torch.randn(1,
    #                     self.embedding_dim, # = D
    #                     self.nb_latents, # = H
    #                     self.nb_latents # = W
    #                     ).repeat(num_steps * num_walks, 1, 1, 1) # = (S x W) = B
    #                     # [B x D x H x W]
        
    #     z_dim = torch.randn((num_steps * num_walks, num_dims)).reshape((num_steps * num_walks, num_dims,1,1)).repeat(1,1,self.nb_latents,self.nb_latents) # [B x d x H x W] (num_dims=d)
    #     index = torch.randint(0, self.embedding_dim, (num_walks, num_dims,)).repeat_interleave(num_steps, dim=0) # [B x d]

    #     z[torch.arange(num_walks*num_steps).repeat_interleave(num_dims).reshape(num_walks*num_steps,num_dims),index,:,:] = z_dim # z[:,d,:,:] = z_dim

    #     z = z.to(current_device)

    #     quantized_inputs, _ = self.vq_layer(z)
    #     samples = self.decode(quantized_inputs)
    #     return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        if "mode" in kwargs and kwargs["mode"] == "causal": # Causes retrieval cannot generate image
            kwargs["mode"] = "action"

        return self.forward(x, **kwargs)[0]

    # def navigate(self, x: Tensor, y: Tensor, steps: int, save_inds: bool = False, **kwargs) -> Tensor:
    #     """
    #     Given an input image x and output image y, returns the intermediate reconstructed images from x to y
    #     :param x: (Tensor) [C x H x W]
    #     :param y: (Tensor) [C x H x W]
    #     :steps: (int) number S-2 of intermediate images
    #     :return: (Tensor) [S x C x H x W]
    #     """
        
    #     # Encode
    #     enc = self.encode(torch.stack([x, y]))[0] # [2 x c x h x w] (c, h, and w are the respective channel, height, and width sizes of the latent space)
    #     enc_shape = enc.shape

    #     # Compute intermediate values
    #     enc_reshaped = enc.reshape(2, enc_shape[1], enc_shape[2]*enc_shape[3], 1).transpose(0,3) # [1 x c x (h x w) x 2]
    #     m=torch.nn.Upsample(size=(enc_shape[2]*enc_shape[3], 2+steps), mode='bilinear', align_corners=True)
    #     enc_reshaped = m(enc_reshaped) # [1 x c x (h x w) x S]
    #     enc_reshaped = enc_reshaped.transpose(0,3).reshape(2+steps, enc_shape[1], enc_shape[2], enc_shape[3]) # [S x c x h x w]

    #     # Decode
    #     quantized_inputs, *encoding_inds = self.vq_layer(enc_reshaped, inds=save_inds)
        
    #     if save_inds:
    #         return self.decode(quantized_inputs), encoding_inds[-1] # [S x C x H x W], [S (x C if using MCQ-VAE) x H x W]
    #     else:
    #         return self.decode(quantized_inputs) # [S x C x H x W]