import torch
import torch_geometric
from models import BaseVAE, MultipleCodebookVectorQuantizer, ResidualLayer
from torch import nn
from torch_geometric import nn as gnn
from torch.nn import functional as F
from .types_ import *
import math


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
                 beta: float = 0.9,
                 **kwargs) -> None:
        super(CausalTransition, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.beta = beta

        self.a_dense = nn.Linear(action_dim, input_dim)

        self.pos_encoding = PositionalEncoding(input_dim)

        self.graph_discovers = nn.ModuleDict({
                "a" : nn.Sequential(
                nn.Linear(3 * input_dim, latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, 1),
                nn.Sigmoid()
                ),

                "y" : nn.Sequential(
                nn.Linear(3 * input_dim, latent_dim),
                nn.LeakyReLU(),
                nn.Linear(latent_dim, 1),
                nn.Sigmoid()
                )
            })

        self.graph_transitioner = gnn.Sequential('x, edge_index', [
                                    (gnn.GATv2Conv(input_dim, latent_dim), 'x, edge_index -> x'),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(latent_dim, input_dim),
                                ])

    # def compute_adj_big(self, latent, action, mode = "a"): # /!\ not tractable!
    #     repeats = latent.size(1)
    #     nodes_i = latent.repeat(1,1,repeats).view((latent.size(0),-1,latent.size(2)))
    #     nodes_j = latent.repeat(1,repeats,1)
        
    #     action = action.view((nodes_i.size(0),1,nodes_i.size(2))).repeat(1,nodes_i.size(1),1).view(nodes_i.shape) # [B x D] --> [B x HW x D]

    #     inp = torch.concat([nodes_i,nodes_j, action],-1)
    #     coeffs = self.graph_discovers[mode](inp).view((-1,repeats,repeats)) # [B x HW x HW]
    #     return coeffs
    
    def compute_adj(self, latent, action, mode = "a"):
        repeats = latent.size(1)
        action = action.unsqueeze(1).repeat(1,repeats,1) # [B x D] --> [B x HW x D]
        coeffs = torch.zeros((latent.size(0), repeats, repeats)).to(device=latent.device)  # [B x HW x HW]

        for i in range(repeats):
            nodes_i = latent[:,i,:].unsqueeze(1).repeat(1,repeats,1)
            nodes_j = latent
            inp = torch.concat([nodes_i, nodes_j, action],-1)
            coeffs[:,i] = self.graph_discovers[mode](inp).view((latent.size(0), repeats))
            
        return coeffs


    def sample_bernoulli(self, adjacency, differentiable=True):
        if differentiable: # reparametrization trick using Straight-through Gumbel-Softmax
            logits = torch.log(torch.stack([1-adjacency,adjacency],dim=-1))
            return F.gumbel_softmax(logits,tau=1,hard=True)[...,1]
        else:
            return torch.bernoulli(adjacency)
    
    def preprocess_nodes_adj(self, latent, action, adjacency, noise=True):
        nodes = torch.concat([latent,action.unsqueeze(1)],1).view((-1,latent.size(-1)))
        if noise:
            nodes = nodes + torch.normal(0,1,nodes.shape).to(latent.device)
        padding = torch.nn.ConstantPad2d((0,1,0,1),1) 
        padded_adjacency = padding(adjacency) # add missing edges to action

        edge_index, _ = torch_geometric.utils.dense_to_sparse(padded_adjacency) # format to edge_index
        return nodes, edge_index # [BHW x D], [2 x E]

    def postprocess_nodes(self, nodes, latent_shape):
        action_latent_shape = list(latent_shape) # [B x HW x D]
        action_latent_shape[1] += 1 # [B x (HW+1) x D]

        return nodes.view((action_latent_shape))[:,:-1,:] # [B x HW x D] (remove action nodes) 

    def infer_action(self, adjacency_coeffs, latent, differentiable=True):
        a = F.one_hot(torch.arange(self.action_dim).repeat(2)).view((latent.size(0),self.action_dim,self.action_dim)).to(latent.device) # [B x A x A]
        a = self.a_dense(a) # [B x A x D]
        a = a.permute(1,0,2) # [A x B x D]

        distances = torch.zeros((a.size(1),self.action_dim)).to(a.device) # [B x A]
        for i in range(self.action_dim):
            adj_i = self.compute_adj(latent, a[i], "a")
            distances[:,i] = F.pairwise_distance(adjacency_coeffs.view((a.size(1), -1)), adj_i.view((a.size(1), -1)))
        probas = 1-torch.softmax(distances,dim=-1)
        
        if differentiable: # reparametrization trick using Straight-through Gumbel-Softmax, adds stochasticity to the process
            logits = torch.log(probas)
            actions = F.gumbel_softmax(logits,tau=1,hard=True)
        else:
            actions = F.one_hot(torch.argmax(probas,dim=1))

        return actions # [B x A]


    def forward(self, latent: Tensor, **kwargs) -> List[Tensor]:
        latent_shape = latent.shape # [B x D x H x W]
        latent = latent.permute(0,2,3,1).view(latent_shape[0], -1, latent_shape[1]) # [B x HW x D]

        pos_latent = self.pos_encoding(latent)  # add positional embeddings
        action = self.a_dense(torch.zeros(latent.size(0), self.action_dim).to(latent.device)) # [B x A]  --> [B x D]

        # Compute causal graph
        causal_graph = self.sample_bernoulli(self.compute_adj(pos_latent, action, "a"))

        # Infer y on causal graph with GNN
        nodes, edge_index = self.preprocess_nodes_adj(pos_latent, action, causal_graph)
        nodes_y = self.graph_transitioner(nodes, edge_index) # [B(HW+1) x D]
        latent_y = self.postprocess_nodes(nodes_y, latent.shape) # [B x HW x D]
        
        latent_y = latent_y.permute(0,2,1).view(latent_shape) # [B x D x H x W]

        # Compute loss function
        id_matrix = F.one_hot(torch.arange(causal_graph.size(-1)).repeat(causal_graph.size(0),1)).to(latent.device, dtype=latent.dtype)
        ct_loss =  F.mse_loss(id_matrix, causal_graph) + F.mse_loss(latent, self.postprocess_nodes(self.graph_transitioner(*self.preprocess_nodes_adj(pos_latent, action, id_matrix)), latent.shape))
        
        return [latent_y, ct_loss]


    def forward_action(self, latent: Tensor, action: Tensor, **kwargs) -> List[Tensor]:
        latent_shape = latent.shape # [B x D x H x W]
        latent = latent.permute(0,2,3,1) # [B x H x W x D]
        latent = latent.view(latent_shape[0], -1, latent_shape[1]) # [B x HW x D]

        pos_latent = self.pos_encoding(latent)  # add positional embeddings
        action = self.a_dense(action) # [B x A] --> [B x D]

        # Compute causal graph
        adjacency_coeffs = self.compute_adj(pos_latent, action, "a")
        causal_graph = self.sample_bernoulli(adjacency_coeffs)

        # Infer y on causal graph with GNN
        nodes, edge_index = self.preprocess_nodes_adj(pos_latent, action, causal_graph)
        nodes_y = self.graph_transitioner(nodes, edge_index) # [B(HW+1) x D]
        latent_y = self.postprocess_nodes(nodes_y, latent.shape) # [B x HW x D]
        
        latent_y = latent_y.permute(0,2,1).view(latent_shape) # [B x H x W x D]

        # Compute loss function
        ct_loss = self.dependencies_MSE_loss(adjacency_coeffs, pos_latent, latent_y, "y") + self.beta * self.graph_size_loss(causal_graph)

        return [latent_y, ct_loss]

    
    def forward_transition(self, latent: Tensor, latent_y: Tensor, **kwargs) -> List[Tensor]:
        latent_shape = latent.shape # [B x D x H x W]
        latent = latent.permute(0,2,3,1) # [B x H x W x D]
        latent = latent.view(latent_shape[0], -1, latent_shape[1]) # [B x HW x D]
        
        pos_latent = self.pos_encoding(latent)  # add positional embeddings
        
        # Compute causal graph
        adjacency_coeffs = self.compute_adj(pos_latent, latent_y, "y")

        # Infer a
        action = self.infer_action(adjacency_coeffs, latent)

        # Compute loss function
        causal_graph = self.sample_bernoulli(adjacency_coeffs)
        ct_loss = self.dependencies_MSE_loss(adjacency_coeffs, pos_latent, self.a_dense(action), "a") + self.beta * self.graph_size_loss(causal_graph)

        return [action, ct_loss]
    
    

    def dependencies_MSE_loss(self, adjacency_coeffs, latent, action, mode = "a"):
        adjacency_coeffs = adjacency_coeffs.detach()
        latent = latent.detach()
        action = action.detach()

        twin_coeffs = self.compute_adj(latent, action, mode)
        return F.mse_loss(twin_coeffs, adjacency_coeffs)
    
    def dependencies_KL_loss(self, causal_graph, latent, action, mode = "a"):
        causal_graph = causal_graph.detach()
        latent = latent.detach()
        action = action.detach()

        twin_cg = self.sample_bernoulli(self.compute_adj(latent, action, mode))
        return F.kl_div(twin_cg, causal_graph)


    def graph_size_loss(causal_graph):
        return torch.linalg.norm(causal_graph)




class CTMCQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 action_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 causal_hidden_dim: int = 800,
                 beta: float = 0.25,
                 causal_beta: float = 0.9,
                 img_size: int = 64,
                 codebooks:int = 1,
                 **kwargs) -> None:
        super(CTMCQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.in_channels = in_channels
        self.beta = beta
        self.codebooks = codebooks

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]
        self.nb_conv = len(hidden_dims)

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

        self.ct_layer = CausalTransition(embedding_dim//codebooks,
                                        action_dim,
                                        causal_hidden_dim,
                                        causal_beta)

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



    def forward_base(self, input: Tensor, **kwargs) -> List[Tensor]:
        # Encoding
        latents = self.encode(input)[0]

        # Causal Transition
        latents_shape = latents.shape
        latents = latents.view((latents_shape[0],latents_shape[1]//self.codebooks,self.codebooks*latents_shape[2],latents_shape[3])) # [B x D x H x W] --> [B x (D/K) x (K*H) x W] (break dimensions into K codebooks and concat blocks in sequence)
        encoding, ct_loss = self.ct_layer(latents) # we may need to apply it either before or after full quantization
        encoding = encoding.reshape(latents_shape) # [B x (D/K) x (K*H) x W] --> [B x D x H x W]

        # Quantization
        quantized_inputs, vq_loss = self.vq_layer(encoding)

        # Decoding
        return [self.decode(quantized_inputs), input, vq_loss, ct_loss]

        
    def forward_action(self, input: Tensor, action: Tensor, input_y: Tensor = None, **kwargs) -> List[Tensor]:
        # Encoding
        latents = self.encode(input)[0]
        
        # Causal Transition
        latents_shape = latents.shape
        latents = latents.view((latents_shape[0],latents_shape[1]//self.codebooks,self.codebooks*latents_shape[2],latents_shape[3])) # [B x D x H x W] --> [B x (D/K) x (K*H) x W]
        encoding, ct_loss = self.ct_layer.forward_action(latents, action)
        encoding = encoding.reshape(latents_shape) # [B x (D/K) x (K*H) x W]--> [B x D x H x W]
        
        # Quantization
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        
        # Decoding
        return [self.decode(quantized_inputs), input_y, vq_loss, ct_loss]

        
    def forward_causal(self, input: Tensor, input_y: Tensor, action: Tensor = None, **kwargs) -> List[Tensor]:
        # Encoding
        latents_x = self.encode(input)[0]
        latents_y = self.encode(input_y)[0]
        
        # Causal Transition
        latents_shape = latents_x.shape
        latents_x = latents_x.view((latents_shape[0],latents_shape[1]//self.codebooks,self.codebooks*latents_shape[2],latents_shape[3])) # [B x D x H x W] --> [B x (D/K) x (K*H) x W]
        latents_y = latents_y.view(latents_x.shape)
        recons_action, ct_loss = self.ct_layer.forward_transition(latents_x, latents_y) # we may need to apply it either before or after full quantization
        
        # Decoding
        return [recons_action, action, 0.0, ct_loss]


    FORWARD_MODES = {
        "base": forward_base,
        "action": forward_action,
        "causal": forward_causal
    }
    def forward(self, input: Tensor, input_y: Tensor = None, action: Tensor = None, mode: str = "base", **kwargs) -> List[Tensor]:
        """
        :param input: (Tensor) Input tensor x to process [N x C x H x W]
        :param input_y: (Tensor) Input tensor y after transition caused by action [N x C x H x W]
        :param action: (Tensor)  Tensor responsible for transition from x to y [N x A]
        :param mode: (str) Mode of application of model.
                "base": encode-decode input
                "action": encode input and decode it after action transition
                "causal": from input and input_y after transition, deduce the actions needed for transition
        """
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

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss + ct_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss': vq_loss,
                'CT_Loss': ct_loss}

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
        nb_latents = self.img_size // 2**self.nb_conv

        z = torch.randn(num_samples,
                        self.embedding_dim,
                        nb_latents,
                        nb_latents) # [B x D x H x W]

        z = z.to(current_device)

        quantized_inputs, _ = self.vq_layer(z)
        samples = self.decode(quantized_inputs)
        return samples

    def walk(self,
               num_steps:int,
               num_dims:int,
               num_walks:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Walks in the latent space and return the corresponding
        images generated at each step. Performs num_walks walks 
        with num_steps steps modifying num_dims dimensions and 
        generates num_walks x num_steps images.
        :param num_steps: (Int) Number of steps in the walk
        :param num_dims: (Int) Number of dimensions to walk along (for a single walk)
        :param num_walks: (Int) Number of walks to perform
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        nb_latents = self.img_size // 2**self.nb_conv

        z = torch.randn(1,
                        self.embedding_dim, # = D
                        nb_latents, # = H
                        nb_latents # = W
                        ).repeat(num_steps * num_walks, 1, 1, 1) # = (S x W) = B
                        # [B x D x H x W]
        
        z_dim = torch.randn((num_steps * num_walks, num_dims)).reshape((num_steps * num_walks, num_dims,1,1)).repeat(1,1,nb_latents,nb_latents) # [B x d x H x W] (num_dims=d)
        index = torch.randint(0, self.embedding_dim, (num_walks, num_dims,)).repeat_interleave(num_steps, dim=0) # [B x d]

        z[torch.arange(num_walks*num_steps).repeat_interleave(num_dims).reshape(num_walks*num_steps,num_dims),index,:,:] = z_dim # z[:,d,:,:] = z_dim

        z = z.to(current_device)

        quantized_inputs, _ = self.vq_layer(z)
        samples = self.decode(quantized_inputs)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def navigate(self, x: Tensor, y: Tensor, steps: int, save_inds: bool = False, **kwargs) -> Tensor:
        """
        Given an input image x and output image y, returns the intermediate reconstructed images from x to y
        :param x: (Tensor) [C x H x W]
        :param y: (Tensor) [C x H x W]
        :steps: (int) number S-2 of intermediate images
        :return: (Tensor) [S x C x H x W]
        """
        
        # Encode
        enc = self.encode(torch.stack([x, y]))[0] # [2 x c x h x w] (c, h, and w are the respective channel, height, and width sizes of the latent space)
        enc_shape = enc.shape

        # Compute intermediate values
        enc_reshaped = enc.reshape(2, enc_shape[1], enc_shape[2]*enc_shape[3], 1).transpose(0,3) # [1 x c x (h x w) x 2]
        m=torch.nn.Upsample(size=(enc_shape[2]*enc_shape[3], 2+steps), mode='bilinear', align_corners=True)
        enc_reshaped = m(enc_reshaped) # [1 x c x (h x w) x S]
        enc_reshaped = enc_reshaped.transpose(0,3).reshape(2+steps, enc_shape[1], enc_shape[2], enc_shape[3]) # [S x c x h x w]

        # Decode
        quantized_inputs, *encoding_inds = self.vq_layer(enc_reshaped, inds=save_inds)
        
        if save_inds:
            return self.decode(quantized_inputs), encoding_inds[-1] # [S x C x H x W], [S (x C if using MCQ-VAE) x H x W]
        else:
            return self.decode(quantized_inputs) # [S x C x H x W]