# models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_Encoder_unit(nn.Module):
    """
    VAE Encoder used in Element-, Coordinate-, and Combined-VAE.
    """
    def __init__(self, z_n, input_shape):
        super(VAE_Encoder_unit, self).__init__()
        data_dim = input_shape[0]
        channel_dim = input_shape[1]
        input_dim = input_shape[2]

        max_filters = 128
        kernel = [5, 3, 3]
        strides = [2, 2, 1]
        padding = [2, 1, 1]
        self.encoder = nn.Sequential(
            nn.Conv1d(channel_dim, max_filters//4, kernel[0], stride=strides[0], padding=padding[0]),
            nn.BatchNorm1d(max_filters//4, momentum=0.01, eps=0.001),
            nn.LeakyReLU(0.2),
            nn.Conv1d(max_filters//4, max_filters//2, kernel[1], stride=strides[1], padding=padding[1]),
            nn.BatchNorm1d(max_filters//2, momentum=0.01, eps=0.001),
            nn.LeakyReLU(0.2),
            nn.Conv1d(max_filters//2, max_filters, kernel[2], stride=strides[2], padding=padding[2]),
            nn.BatchNorm1d(max_filters, momentum=0.01, eps=0.001),
            nn.LeakyReLU(0.2),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(math.ceil(input_dim/4) * max_filters, 1024)
        self.sigmoid = nn.Sigmoid()
        self.fc_mean = nn.Linear(1024, z_n)
        self.fc_log_var = nn.Linear(1024, z_n)
        self.double()

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var

    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)


class VAE_Encoder_Fully_connected_unit(nn.Module):
    """
    Fully connected encoder used in Lattice-VAE.
    """
    def __init__(self, z_n, input_shape):
        super(VAE_Encoder_Fully_connected_unit, self).__init__()
        data_dim = input_shape[0]
        channel_dim = input_shape[1]
        input_dim = input_shape[2]
        data_size = channel_dim * input_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(data_size, 5),
            nn.Sigmoid(),
        )
        self.fc_mean = nn.Linear(5, z_n)
        self.fc_log_var = nn.Linear(5, z_n)
        self.double()

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        z = self.reparameterize(mean, log_var)
        return z, mean, log_var

    @staticmethod
    def reparameterize(mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)


class VAE_Decoder_unit(nn.Module):
    """
    Decoder used in Element-, Coordinate-, and Combined-VAE.
    """
    def __init__(self, z_n, input_shape):
        super(VAE_Decoder_unit, self).__init__()
        data_dim = input_shape[0]
        channel_dim = input_shape[1]
        input_dim = input_shape[2]
        self.map_size = input_dim // 4
        self.max_filters = 128
        kernel = [5, 3, 3]
        strides = [2, 2, 1]
        padding = [2, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(z_n, self.max_filters * self.map_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(self.max_filters, momentum=0.01, eps=0.001),
            nn.ConvTranspose1d(self.max_filters, self.max_filters//2, kernel[2],
                                stride=strides[2], padding=padding[2]),
            nn.BatchNorm1d(self.max_filters//2, momentum=0.01, eps=0.001),
            nn.ConvTranspose1d(self.max_filters//2, self.max_filters//4, kernel[1],
                                stride=strides[1], padding=padding[1], output_padding=1),
            nn.BatchNorm1d(self.max_filters//4, momentum=0.01, eps=0.001),
            nn.ConvTranspose1d(self.max_filters//4, channel_dim, kernel[0],
                                stride=strides[0], padding=padding[0], output_padding=1),
            nn.Sigmoid(),
        )
        self.double()

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.max_filters, self.map_size)
        x_hat = self.decoder(x)
        return x_hat


class VAE_Decoder_Fully_connected_unit(nn.Module):
    """
    Fully connected decoder used in Lattice-VAE.
    """
    def __init__(self, z_n, input_shape):
        super(VAE_Decoder_Fully_connected_unit, self).__init__()
        data_dim = input_shape[0]
        self.channel_dim = input_shape[1]
        self.input_dim = input_shape[2]
        data_size = self.channel_dim * self.input_dim
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(z_n, 5),
            nn.ReLU(),
            nn.Linear(5, data_size),
        )
        self.double()

    def forward(self, z):
        x = self.fc(z)
        x_hat = x.reshape(-1, self.channel_dim, self.input_dim)
        return x_hat


class Predictor(nn.Module):
    """
    Property predictor module.
    """
    def __init__(self, input_dim, regression_dim):
        super(Predictor, self).__init__()
        self.regression = nn.Sequential(
            nn.ReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 8),
            nn.ReLU(),
            nn.Linear(input_dim // 8, regression_dim),
            nn.Sigmoid(),
        )
        self.double()

    def forward(self, z_mean):
        y_hat = self.regression(z_mean)
        return y_hat


class VAE_Lattice(nn.Module):
    """
    Lattice-VAE: connects the fully connected encoder and decoder.
    """
    def __init__(self, z_n, x, y):
        super(VAE_Lattice, self).__init__()
        input_shape_x = x.shape
        self.encoder = VAE_Encoder_Fully_connected_unit(z_n, input_shape_x)
        self.decoder = VAE_Decoder_Fully_connected_unit(z_n, input_shape_x)

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = None
        return z, x_hat, mean, log_var, y_hat


class VAE(nn.Module):
    """
    VAE connecting the convolutional encoder/decoder and property predictor.
    """
    def __init__(self, z_n, x, y):
        super(VAE, self).__init__()
        input_shape_x = x.shape
        input_shape_y = y.shape
        self.encoder = VAE_Encoder_unit(z_n, input_shape_x)
        self.decoder = VAE_Decoder_unit(z_n, input_shape_x)
        self.regression = Predictor(input_dim=z_n, regression_dim=input_shape_y[1])

    def forward(self, x):
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.regression(mean)
        return z, x_hat, mean, log_var, y_hat

