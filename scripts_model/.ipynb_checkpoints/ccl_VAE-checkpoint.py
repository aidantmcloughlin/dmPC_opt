import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.utils
import torch.distributions
import numpy as np

from copy import deepcopy


class AE(nn.Module):
    def __init__(self,
                 input_dim,
                 h_dims=[512],
                 latent_dim = 128,
                 drop_out=0):                
        super(AE, self).__init__()

        self.latent_dim = latent_dim

        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)

        # Encoder
        modules_e = []
        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules_e.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU())
                )

        self.encoder_body = nn.Sequential(*modules_e)
        self.encoder_last_layer = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        hidden_dims.reverse()

        modules_d = []

        self.decoder_first_layer = nn.Linear(latent_dim, hidden_dims[1])

        for i in range(len(hidden_dims) - 2):
            i_dim = hidden_dims[i]
            o_dim = hidden_dims[i + 1]

            modules_d.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU())
            )
        
        self.decoder_body = nn.Sequential(*modules_d)

        self.decoder_last_layer = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU() # Sigmoid()
        )
            

    def encode(self, X: Tensor):
        """
        Encodes the inputed tensor X by passing through the encoder network.
        Returns the latent space Z (the encoded X) as a tensor.
        """
        result = self.encoder_body(X)
        Z = self.encoder_last_layer(result)

        return Z

    def decode(self, Z: Tensor):
        """
        Decodes the inputed tensor Z by passing through the decoder network.
        Returns the reconstructed X (the decoded Z) as a tensor. 
        """
        result = self.decoder_first_layer(Z)
        result = self.decoder_body(result)
        X_rec = self.decoder_last_layer(result)

        return X_rec

    def forward(self, X: Tensor):
        """
        Passes X through the encoder and decoder networks.
        Returns the reconstructed X. 
        """
        Z = self.encode(X)
        X_rec = self.decode(Z)

        return X_rec

       
class VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 h_dims=[512],
                 latent_dim = 128,
                 drop_out=0):                
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, input_dim)

        # Encoder
        modules_e = []
        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules_e.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.ReLU())
                )

        self.encoder_body = nn.Sequential(*modules_e)
        self.encoder_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        hidden_dims.reverse()

        modules_d = []

        self.decoder_first_layer = nn.Linear(latent_dim, hidden_dims[1])

        for i in range(len(hidden_dims) - 2):
            i_dim = hidden_dims[i]
            o_dim = hidden_dims[i + 1]

            modules_d.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.ReLU())
            )
        
        self.decoder_body = nn.Sequential(*modules_d)

        self.decoder_last_layer = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU() # Sigmoid()
        )
            

    def encode_(self, X: Tensor):
        """
        Encodes the inputed tensor X by passing through the encoder network.
        Return a list with two tensors, mu and log_variance of the latent space. 
        """
        result = self.encoder_body(X)
        mu = self.encoder_mu(result)
        log_var = self.encoder_logvar(result)
       
        return [mu, log_var]

    def encode(self, X: Tensor, repram=True):
        """
        Encodes the inputed tensor X by passing through the encoder network.
        Returns the reparameterized latent space Z if reparameterization == True;
        otherwise returns the mu of the latent space.
        """
        mu, log_var = self.encode_(X)

        if(repram==True):
            Z = self.reparameterize(mu, log_var)
            return Z
        else: 
            return mu

    def decode(self, Z: Tensor):
        """
        Decodes the inputed tensor Z by passing through the decoder network.
        Returns the reconstructed X (the decoded Z) as a tensor. 
        """
        result = self.decoder_first_layer(Z)
        result = self.decoder_body(result)
        X_rec = self.decoder_last_layer(result)

        return X_rec

    def forward(self, X: Tensor):
        """
        Passes X through the encoder and decoder networks.
        Returns the reconstructed X. 
        """
        mu, log_var = self.encode_(X)
        Z = self.reparameterize(mu, log_var)
        X_rec = self.decode(Z)

        return [X, mu, log_var, Z, X_rec]
    
    def reparameterize(self, mu: Tensor, log_var: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return z



def idx2onehot(idx, n):

    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)
    
    return onehot

class CVAE(VAE):
    def __init__(self,
                 input_dim,
                 n_conditions,
                 h_dims=[512],
                 latent_dim = 128,
                 drop_out=0):                
        
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_conditions = n_conditions


        hidden_dims = deepcopy(h_dims)
        hidden_dims.insert(0, self.input_dim + self.n_conditions)

        # Encoder
        modules_e = []
        for i in range(1, len(hidden_dims)):
            i_dim = hidden_dims[i-1]
            o_dim = hidden_dims[i]

            modules_e.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU()
                    )
                )

        self.encoder_body = nn.Sequential(*modules_e)
        self.encoder_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        hidden_dims.reverse()
        hidden_dims[-1] = self.input_dim 

        modules_d = []

        self.decoder_first_layer = nn.Linear(latent_dim+self.n_conditions, hidden_dims[1])

        for i in range(len(hidden_dims) - 2):
            i_dim = hidden_dims[i]
            o_dim = hidden_dims[i + 1]

            modules_d.append(
                nn.Sequential(
                    nn.Linear(i_dim, o_dim),
                    nn.BatchNorm1d(o_dim),
                    nn.Dropout(drop_out),
                    nn.LeakyReLU())
            )
        
        self.decoder_body = nn.Sequential(*modules_d)

        self.decoder_last_layer = nn.Sequential(
            nn.Linear(hidden_dims[-2], hidden_dims[-1]),
            nn.ReLU() # Sigmoid()
        )
            

    def encode_(self, X: Tensor, C: Tensor):
        """
        Encodes the inputed tensor X by passing through the encoder network.
        Return a list with two tensors, mu and log_variance of the latent space. 
        """
        C = idx2onehot(C, n=self.n_conditions)
        X_C = torch.cat((X,C), dim=-1)

        result = self.encoder_body(X_C)
        mu = self.encoder_mu(result)
        log_var = self.encoder_logvar(result)
       
        return [mu, log_var]

    def encode(self, X: Tensor, repram=True):
        """
        Encodes the inputed tensor X by passing through the encoder network.
        Returns the reparameterized latent space Z if reparameterization == True;
        otherwise returns the mu of the latent space.
        """
        mu, log_var = self.encode_(X)

        if(repram==True):
            Z = self.reparameterize(mu, log_var)
            return Z
        else: 
            return mu

    def decode(self, Z: Tensor, C: Tensor):
        """
        Decodes the inputed tensor Z by passing through the decoder network.
        Returns the reconstructed X (the decoded Z) as a tensor. 
        """
        C = idx2onehot(C, n=self.n_conditions)
        Z_C = torch.cat((Z, C), dim=-1)

        result = self.decoder_first_layer(Z_C)
        result = self.decoder_body(result)
        X_rec = self.decoder_last_layer(result)

        return X_rec

    def forward(self, X: Tensor, C: Tensor):
        """
        Passes X through the encoder and decoder networks.
        Returns the reconstructed X. 
        """
        mu, log_var = self.encode_(X, C)
        Z = self.reparameterize(mu, log_var)
        X_rec = self.decode(Z, C)

        return [X, mu, log_var, Z, X_rec]
    
    def reparameterize(self, mu: Tensor, log_var: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return z
