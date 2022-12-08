import torch
from models import BaseVAE, ResidualLayer
from torch import nn
from torch.nn import functional as F
from .types_ import *

class VectorQuantizerMS(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    Vector quantizer implementation with Multiple Steps. Computation of indicator encodings is separated from quantization step.
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        super(VectorQuantizerMS, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)


    def compute_inds(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        return encoding_inds.view(latents_shape[:3]) # [B x H x W]

    def compute_latents(self, latents: Tensor, encoding_inds: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape

        # Convert to one-hot encodings
        device = latents.device
        encoding_inds = encoding_inds.reshape((-1, 1))
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss


    def forward(self, latents: Tensor, inds : bool = False) -> Tensor:
        encoding_inds = self.compute_inds(latents)
        quantized_latents, vq_loss = self.compute_latents(latents, encoding_inds)

        if inds:
            return quantized_latents, vq_loss, encoding_inds  # [B x D x H x W], _, # [B x H x W]
        else:
            return quantized_latents, vq_loss  # [B x D x H x W]



class MultipleCodebookVectorQuantizer(nn.Module):
    """
        VectorQuantizer with a number C of codebooks sharing the embedding_dim between them
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 codebooks: int,
                 beta: float = 0.25):
        super(MultipleCodebookVectorQuantizer, self).__init__()

        assert embedding_dim % codebooks == 0 # embedding size must be divided between all codebooks

        self.nb_codebooks = codebooks
        self.reduced_embedding_dim = embedding_dim // codebooks

        self.quantizers = nn.ModuleList([VectorQuantizerMS(num_embeddings,
                                                    self.reduced_embedding_dim,
                                                    beta) 
                                        for _ in range(codebooks)])


    def compute_inds(self, latents: Tensor) -> Tensor:
        encoding_inds = []

        for i, quantizer in enumerate(self.quantizers):
            sublatents = latents[:,i:i+self.reduced_embedding_dim,:,:]  # [B x D x H x W]
            encoding_ind = quantizer.compute_inds(sublatents)
            encoding_inds.append(encoding_ind)
        
        encoding_inds = torch.stack(encoding_inds, 1)

        return encoding_inds # [B x C x H x W]
    
    def compute_latents(self, latents: Tensor, encoding_inds: Tensor) -> Tensor:
        quantized_latents = []
        vq_loss = []

        for i, quantizer in enumerate(self.quantizers):
            sublatents = latents[:,i:i+self.reduced_embedding_dim,:,:]  # [B x D x H x W]
            subinds = encoding_inds[:,i,:,:].squeeze(1) # [B x C x H x W] -> [B x H x W]
            quantized_sublatents, vq_subloss = quantizer.compute_latents(sublatents, subinds)

            quantized_latents.append(quantized_sublatents)
            vq_loss.append(vq_subloss)
        
        quantized_latents = torch.cat(quantized_latents, 1)
        vq_loss = sum(vq_loss)

        return quantized_latents, vq_loss


    def forward(self, latents: Tensor, inds : bool = False) -> Tensor:
        encoding_inds = self.compute_inds(latents)
        quantized_latents, vq_loss = self.compute_latents(latents, encoding_inds)

        if inds:
            return quantized_latents, vq_loss, encoding_inds  # [B x D x H x W], _, # [B x H x W]
        else:
            return quantized_latents, vq_loss  # [B x D x H x W]




class MCQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                 hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 codebooks:int = 1,
                 **kwargs) -> None:
        super(MCQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.in_channels = in_channels
        self.beta = beta

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

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

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

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

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


    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

