"""
Code for Crystal-LCA-LSBO experiments.
We provided the trained models and other required files.
Note that you need to create API key from Materials Project to run this code. 
After creating this key, please copy it to line 108, and follow the intructions in README file.
"""

import os
import pickle
import sys
import joblib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.preprocessing import OneHotEncoder
# from pymatgen.core import structure
# from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from ase.io import write
from ase import spacegroup
import argparse
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.acquisition import AcquisitionFunction
import xgboost as xgb
from gpytorch.kernels import ScaleKernel, RBFKernel
import gpytorch
from models import *
import subprocess
import re
import os
from data_utils import minmax_, inv_minmax, data_query, FTCP_represent, pad, convert_cif
from dataset import MyDataset

# in case of GPU-usage, below functions automatically detects the emptiest GPU and sets visible devices accordingly
"""
def get_gpu_memory_usage():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE, universal_newlines=True)
    gpu_memory_info = [line.split(',') for line in result.stdout.strip().split('\n')]
    return [(int(used), int(total)) for used, total in gpu_memory_info]

def find_empty_gpu():
    memory_info = get_gpu_memory_usage()
    
    if not memory_info:
        print("No GPUs found.")
        return None

    # Find the GPU with the most available memory
    print(memory_info)
    empty_gpu = min(range(len(memory_info)), key=lambda i: memory_info[i][0] / memory_info[i][1])

    return empty_gpu

# Example usage:
empty_gpu_index = find_empty_gpu()

if empty_gpu_index is not None:
    print(f"The emptiest GPU is GPU {empty_gpu_index}.")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(empty_gpu_index)
    # Now, you can use the selected GPU for your task.
else:
    print("No available GPU found.")
"""

parser = argparse.ArgumentParser(description='exp name')

parser.add_argument('combined_z_size', type=int, help='The digit value')
parser.add_argument('ckpt_name', type=str, help='The digit value')
parser.add_argument('bound', type=float, help='Center of the condition vector')
parser.add_argument('gamma', type=float, help='Center of the condition vector')
parser.add_argument('roi_var', type=float, help='Center of the condition vector')
parser.add_argument('threshold', type=float, help='Center of the condition vector')

args = parser.parse_args()

combined_z_size = args.combined_z_size
ckpt_name = args.ckpt_name
bound = args.bound
gamma = args.gamma
roi_var = args.roi_var
input_threshold = args.threshold
experiment_name = 'crystal_lca_lsbo'

element_z_size = 16
coord_z_size = 16


#ーーーーーーーーーーーーーーーーparameterーーーーーーーーーーーーーーーーーーー
# parameter of query materials
max_elms = 4
min_elms = 3
max_sites = 40
# Use your own API key to query Materials Project 
mp_api_key = '' 
epochs = 500
lr_0 = 5e-4
batch_size = 256
loss_coeff=(5, 10, 1)
property_for_pridict = ['formation_energy_per_atom', 'band_gap',]
exp_name = "crystal_lsbo_repo"
exp_name2 = "crystal-lca-lsbo"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')
#torch.cuda.empty_cache()
#ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー


def loss_function(x, x_hat, mean, log_var, y, y_hat, loss_coeff=[1, 1, 1]):
    """
    Loss function used in training

    Parameters
    ----------
    x            -> input data
    x_hat        -> reconstructed data
    mean         -> mean parameter of distribution in latent space
    log_var      -> log variance parameter of distribution in latent space
    y            -> input property
    y_hat        -> predicted property
    loss_coeff   -> weights for reconstruction, kl divergence, and property prediction parts

    
    Returns
    ---------- 
    elbo_loss -> VAE loss
    reconstruction_loss -> reconstruction loss part of the VAE loss
    kld_loss -> KL divergence loss part of the VAE loss
    """
    # ELBO + Property Loss
    batch, channel, feature = x.shape
    mse_loss_fn = nn.MSELoss(reduction='sum')
    reconstruction_loss = mse_loss_fn(x, x_hat) / channel
    # KLD loss
    kld_loss = - 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
    
    if y != None and y_hat != None:
        mse_loss_fn_y = nn.MSELoss(reduction='sum')
        prop_pred_loss = mse_loss_fn_y(y, y_hat)
        elbo_loss = torch.mean(
            loss_coeff[0] * reconstruction_loss +  loss_coeff[1] * kld_loss + loss_coeff[2] * prop_pred_loss
        )
        return elbo_loss, reconstruction_loss, kld_loss, prop_pred_loss
    else:
        elbo_loss = torch.mean(
            loss_coeff[0] * reconstruction_loss +  loss_coeff[1] * kld_loss
        )
        return elbo_loss, reconstruction_loss, kld_loss, torch.tensor(0)


def concat_roi_to_dataset(model, roi_set, trainloader, train_dataset):
    """
    Concatenates the instance generated at the region of interest to the training dataset    

    Parameters
    ----------
    model -> VAE model
    roi_set -> latent variable at region of interest
    trainloader -> trainloder of the model
    train_dataset -> training dataset of the model

    Returns
    ---------- 
    trainloader -> trainloder of the model
    """
    # concatenate generation from region of interest to the training dataset
    with torch.no_grad():
        gen_structures = model.decoder.forward(roi_set.to(device).to(torch.float64))
        _, _, _, _, y_hat = model.forward(gen_structures)
    if y_hat is None:
        data = [[0.9, 0.1]]
        # Create the tensor with the specified dtype
        y_hat = torch.tensor(data, dtype=torch.float64)
    train_dataset.dataset.add_instance(torch.Tensor(gen_structures), y_hat)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # num_workers=2
    return trainloader

def retrain_with_roi(model,roi, retraining_epoch, trainloader, gamma):
    """
    Retrains Lattice, Coordinate, Element-VAE models with data augmentations in latent space

    Parameters
    ----------
    model -> VAE model
    roi   -> region of interest (mean of p^{ref})
    retraining_epoch -> number of epochs in retraining
    trainloader -> trainloder of the model
    gamma -> weight for the penalization term of reconstructions of augmented latent variables

    Returns
    ---------- 
    None
    """
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    z_size = roi.shape[1]
    for _ in range(0, retraining_epoch):
        train_loss=0.0
        train_reconst_loss=0.0
        train_kld_loss=0.0
        train_prop_loss=0.0
        test_loss=0.0
        test_reconst_loss=0.0
        test_kld_loss=0.0
        test_prop_loss=0.0
        train_latent_recon_loss = 0.0
        test_latent_recon_loss = 0.0
        train_latent_recon_loss_art = 0.0
        test_latent_recon_loss_art = 0.0
        loss_coeff=(1, 0.01, 0)
        model.train()
        for i, (x, y) in enumerate(trainloader):
            optimizer.zero_grad()
            x_train = x.to(device).to(torch.float64)
            y_train = y.to(device)
            _, x_train_hat, mean, log_var, y_hat = model(x_train)
            if y_hat is None:
                data = [[0.9, 0.1]]
                # Create the tensor with the specified dtype
                y_hat = torch.tensor(data, dtype=torch.float64).to(device)
            _, x_train_hat_latent_recon, mean_latent_recon, _, _ = model(x_train_hat)
            latent_recon_error = nn.MSELoss()(mean, mean_latent_recon)


            z_sample = torch.normal(0, 1, [batch_size, z_size]).to(torch.float64).to(device) * roi_var + roi.to(device)
            gen_structures = model.decoder.forward(z_sample.to(device))
            _, _, z_hat, _, _ = model(gen_structures)
            latent_recon_error_art = nn.MSELoss()(z_sample.to(device), z_hat.to(device))

            elbo_loss, reconst_loss, kld_loss, prop_loss = loss_function(x_train.to(torch.float32), x_train_hat.to(torch.float32), mean.to(torch.float32), log_var.to(torch.float32), y_train.to(torch.float32), y_hat.to(torch.float32), loss_coeff)
            elbo_loss += gamma * latent_recon_error_art 
            elbo_loss.backward()
            optimizer.step()

            train_latent_recon_loss += latent_recon_error.item()
            train_latent_recon_loss_art += latent_recon_error_art.item()
            train_loss += elbo_loss.item()
            train_reconst_loss += reconst_loss.item()
            train_kld_loss += kld_loss.item()
            train_prop_loss += prop_loss.item()

# data.csv will include the CIF and property information
try:
    dataframe = pd.read_csv(f"{exp_name}/data.csv")
except:
    dataframe = data_query(mp_api_key, max_elms, min_elms, max_sites)
    dataframe.to_csv(f"{exp_name}/data.csv")

try:
    # we use the same data as FTCP-VAE paper, but we will omit their fourier-transformed features below
    with open('{}/FTCP_representation.pkl'.format(exp_name), 'rb') as file:
        FTCP_representation = pickle.load(file)
    with open('{}/Nsites.pkl'.format(exp_name), 'rb') as file:
        Nsites = pickle.load(file)
except:
    FTCP_representation, Nsites = FTCP_represent(dataframe, max_elms, max_sites, return_Nsites=True)
    FTCP_representation = pad(FTCP_representation, 2)
    with open('{}/FTCP_representation.pkl'.format(exp_name), 'wb') as file:
        pickle.dump(FTCP_representation, file)
    with open('{}/Nsites.pkl'.format(exp_name), 'wb') as file:
        pickle.dump(Nsites, file)

FTCP_representation = pad(FTCP_representation, 2)
print("FTCP_representation:", FTCP_representation.shape)
X_array_origin = FTCP_representation
del FTCP_representation
prop = property_for_pridict
Y_array = dataframe[prop].values
del dataframe

torch.manual_seed(42)
np.random.seed(42)

# omit fourier transformed features and use only crystal structure features
subset_1 = X_array_origin[:, 0:103, :4] # element
subset_2 = X_array_origin[:, 103:104, :3] # angle
subset_3 = X_array_origin[:, 104:105, :3] # abc
subset_4 = X_array_origin[:, 105:145, :4] # coor
subset_5 = X_array_origin[:, 145:185, :4] # occup
subset_6 = X_array_origin[:, 185:, :] # propert


### load element model ###
X_array = np.concatenate((subset_1, subset_5), axis=1) # element model uses element and occupancy parts
del subset_1, subset_5
input_normed, Y_normed, scaler_x_element, scaler_y_element = minmax_(X_array, Y_array)
input_X_full = torch.tensor(input_normed.transpose(0, 2, 1)).double()
input_Y = torch.tensor(Y_normed).double()
input_X_element = f.pad(input_X_full[:, :4, :143], (0, 1), 'constant', 0)
num_of_dataset = len(input_X_element)
num_of_trainset = 40000
num_of_testset = num_of_dataset - num_of_trainset
dataset_element = MyDataset(input_X_element, input_Y)
generator1 = torch.Generator().manual_seed(42)
train_dataset_element, test_dataset_element = random_split(dataset_element, [num_of_trainset, num_of_testset], generator=generator1)
trainloader_element = DataLoader(train_dataset_element, batch_size=batch_size, shuffle=True)
testloader_element = DataLoader(test_dataset_element, batch_size=batch_size, shuffle=True)
model_element = VAE(element_z_size, input_X_element, input_Y).to(device).to(torch.float64)
model_element.load_state_dict(torch.load(f'crystal_lsbo_repo/element_vae.pt', map_location=torch.device(device)))

_, _, mean_element_train, _, _ = model_element.forward(dataset_element[train_dataset_element.indices][0].to(device))
_, hop, mean_element_test, _, _ = model_element.forward(dataset_element[test_dataset_element.indices][0].to(device))
print(mean_element_train.shape)
print(mean_element_test.shape)

### load lattice model ###
lattice_z_size = 3
X_array = np.concatenate((subset_2, subset_3), axis=1) # combine angle and cell lengths
del subset_2, subset_3
X_array = X_array[:, :, :4]
input_normed, Y_normed, scaler_x_lattice, scaler_y_lattice = minmax_(X_array, Y_array)
input_X_full = torch.tensor(input_normed.transpose(0, 2, 1)).double()
input_Y = torch.tensor(Y_normed).double()
input_X_lattice = input_X_full[:, :4, :]
num_of_dataset = len(input_X_lattice)
num_of_trainset = 40000
num_of_testset = num_of_dataset - num_of_trainset

dataset_lattice = MyDataset(input_X_lattice, input_Y)
train_dataset_lattice, test_dataset_lattice = random_split(dataset_lattice, [num_of_trainset, num_of_testset],generator=generator1)
trainloader_lattice = DataLoader(train_dataset_lattice, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset_lattice, batch_size=batch_size, shuffle=True)

model_lattice = VAE_Lattice(lattice_z_size, input_X_lattice, input_Y).to(device).to(torch.float64)
model_lattice.load_state_dict(torch.load('{}/lattice_vae.pt'.format(exp_name), map_location=torch.device(device)))

_, _, mean_lattice_train, _, _ = model_lattice.forward(dataset_lattice[train_dataset_lattice.indices][0].to(device))
_, hop_lattice, mean_lattice_test, _, _ = model_lattice.forward(dataset_lattice[test_dataset_lattice.indices][0].to(device))


### load coordinate model ###
X_array = subset_4
del subset_4
input_normed, Y_normed, scaler_x_coor, scaler_y_coor = minmax_(X_array, Y_array)
input_X_full = torch.tensor(input_normed.transpose(0, 2, 1)).double()
input_Y = torch.tensor(Y_normed).double()
input_X_coord = input_X_full[:, :4, :]
del input_X_full, input_normed, Y_normed
num_of_dataset = len(input_X_coord)
num_of_trainset = 40000
num_of_testset = num_of_dataset - num_of_trainset
dataset_coord = MyDataset(input_X_coord, input_Y)
train_dataset_coord, test_dataset_coord = random_split(dataset_coord, [num_of_trainset, num_of_testset],generator=generator1)
trainloader_coord = DataLoader(train_dataset_coord, batch_size=batch_size, shuffle=True)
testloader_coord = DataLoader(test_dataset_coord, batch_size=batch_size, shuffle=True)
model_coord = VAE(coord_z_size, input_X_coord, input_Y).to(device).to(torch.float64)
model_coord.load_state_dict(torch.load('{}/coordinate_vae.pt'.format(exp_name), map_location=torch.device(device)))

_, _, mean_coord_train, _, _ = model_coord.forward(dataset_coord[train_dataset_coord.indices][0].to(device))
_, _, mean_coord_test, _, _ = model_coord.forward(dataset_coord[test_dataset_coord.indices][0].to(device))


train_set = torch.cat((mean_element_train, mean_lattice_train, mean_coord_train), dim=1)
test_set = torch.cat((mean_element_test, mean_lattice_test, mean_coord_test), dim=1)


input_X_comb = torch.load('crystal_lsbo_repo/combined_vae_input_latent_data.pt') 
del train_set

### LOAD COMBINED MODEL ####
data_min = input_X_comb.min()
data_max = input_X_comb.max()
input_X_comb = f.pad(input_X_comb, (0, 1), 'constant', 0)
input_X_comb = input_X_comb.view(input_X_comb.shape[0], 3, 12)
num_of_dataset = len(input_X_comb)
num_of_trainset = 1
num_of_testset = num_of_dataset - num_of_trainset
dataset = MyDataset(input_X_comb, input_Y)
train_dataset, test_dataset = random_split(dataset, [num_of_trainset, num_of_testset],generator=generator1)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
model_combined = VAE(combined_z_size, input_X_comb, input_Y).to(device).to(torch.float64)
model_combined.load_state_dict(torch.load('{}/{}'.format(exp_name,ckpt_name), map_location=torch.device(device)))

def rescale_and_divide(x_recon):
    """
    Rescales the Combined-VAE generations and divide it into Lattice-VAE, Coordinate-VAE, 
    and Element-VAE inputs

    Parameters
    ----------
    x_recon -> generated latent variables from Combined-VAE

    
    Returns
    ---------- 
    element_z -> latent variable for Element-VAE
    lattize_z -> latent variable for Lattice-VAE
    coord_z -> latent variable for Lattice-VAE
    """
    x_recon = x_recon.view(-1, 36)
    x_recon = x_recon * (data_max - data_min) + data_min
    element_z = x_recon[:,:element_z_size]
    lattize_z = x_recon[:,element_z_size:(element_z_size+3)]
    coord_z = x_recon[:,(element_z_size+3):(element_z_size+3+coord_z_size)]
    return element_z, lattize_z, coord_z


def get_lattice_generation(lattize_z_sample_first, Y_test_array):
    """
    Gets lattice generation given lattize_z_sample_first

    Parameters
    ----------
    lattize_z_sample_first -> lattice latent variable generated from Combined-VAE, input to Lattice-VAE
    Y_test_array -> dummy variable for Y to be used in inv_minmax function call

    
    Returns
    ---------- 
    lattize_z_sample -> generated lattice from Lattice-VAE
    """
    lattize_z_sample = model_lattice.decoder.forward(lattize_z_sample_first.to(device).to(torch.float64))
    lattize_z_sample = torch.Tensor(lattize_z_sample).to(device)
    lattize_z_sample = lattize_z_sample.detach().cpu().numpy()
    lattize_z_sample,_ = inv_minmax(lattize_z_sample, Y_test_array, scaler_x_lattice, scaler_y_lattice)
    return lattize_z_sample


def get_coor_generation(coord_z_sample_first, Y_test_array):
    """
    Gets coordinate generation given coord_z_sample_first

    Parameters
    ----------
    coord_z_sample_first -> coordinate latent variable generated from Combined-VAE, input to Coordinate-VAE
    Y_test_array -> dummy variable for Y to be used in inv_minmax function call

    
    Returns
    ---------- 
    coord_z_sample -> generated coordinate from Coordinate-VAE
    """    
    coord_z_sample = model_coord.decoder.forward(coord_z_sample_first.to(device).to(torch.float64))
    coord_z_sample = torch.Tensor(coord_z_sample).to(device)
    coord_z_sample = coord_z_sample.detach().cpu().numpy()
    coord_z_sample,_ = inv_minmax(coord_z_sample, Y_test_array, scaler_x_coor, scaler_y_coor)
    return coord_z_sample


subset_1 = X_array_origin[:, 0:103, :]
subset_5 = X_array_origin[:, 145:185, :]
X_array = np.concatenate((subset_1, subset_5), axis=1)

X_normed, Y_normed, scaler_x, scaler_y = minmax_(X_array, Y_array)
input_X = torch.tensor(X_normed.transpose(0, 2, 1)).double()
input_Y = torch.tensor(Y_normed).double()

num_of_dataset = len(input_X)
num_generate = 1000
num_of_test_data = 1000
indices = torch.randperm(num_of_dataset)[:num_of_test_data]
X_test = input_X[indices].to(device)
Y_test = input_Y[indices].to(device)

def get_element_generation(element_z_sample_first):
    """
    Gets element generation given coord_z_sample_first

    Parameters
    ----------
    element_z_sample_first -> element latent variable generated from Combined-VAE, input to Element-VAE
    Y_test_array -> dummy variable for Y to be used in inv_minmax function call

    
    Returns
    ---------- 
    element_z_sample_first -> generated element from Coordinate-VAE
    """    
    with torch.no_grad():
        element_z_gen = model_element.decoder.forward(element_z_sample_first.to(device).to(torch.float64))
                    
    Y_test_array = Y_test.detach().cpu().numpy()
    element_z_gen = element_z_gen.detach().cpu().numpy()
        
    X_test_gen_array = np.zeros((1, X_test.shape[1], X_test.shape[2]))
            
    X_test_gen_array[:, :4, 0:143] = element_z_gen[:, :4, 0:143]
    X_test_gen_array, Y_test_array_origin = inv_minmax(X_test_gen_array, Y_test_array, scaler_x, scaler_y)
    return X_test_gen_array, Y_test_array

def get_updated_embeddings(model, dataloader):
    """
    After Lattice-VAE, Coordinate-VAE, Element-VAE models are retrained,
    we need to obtain the updated latent representations of training data in order to
    retrain the Combined-VAE.
    This function gets these updated latent representations (embeddings) for each model 

    Parameters
    ----------
    model -> VAE model (lattice-coordinate-element)
    dataloader -> dataloader of the VAE model

    
    Returns
    ---------- 
    embeddings -> new embeddings for respective parts of the crystals
    """
    embeddings = torch.Tensor()
    with torch.no_grad():
        for _, (x, _) in enumerate(dataloader):
            _, _, embedding, _, _ = model.forward(x)
            embeddings = torch.cat([embeddings, embedding])
    return embeddings

def comb_train(model, optimizer, epochs, trainloader, loss_coeff=[1, 1, 1]):
    """
    Retrains the combined-vae model

    Parameters
    ----------
    model -> Combined-VAE model
    optimizer -> optimizer used in training
    epochs -> number of epochs
    trainloader -> trainloader of Combined-VAE
    loss_coeff   -> weights for reconstruction, kl divergence, and property prediction parts


    
    Returns
    ---------- 
    None
    """
    reduce_lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=4, min_lr=1e-6)
    for epoch in range(1, epochs+1):
        train_loss=0.0
        train_reconst_loss=0.0
        train_kld_loss=0.0
        train_prop_loss=0.0
        model.train()
        for i, (x, y) in enumerate(trainloader):
            optimizer.zero_grad()
            x_train = x.to(device)
            y_train = y.to(device)
            _, x_train_hat, mean, log_var, y_hat = model(x_train)
            elbo_loss, reconst_loss, kld_loss, prop_loss = loss_function(x_train, x_train_hat, mean, log_var, y_train, y_hat, loss_coeff)
            elbo_loss.to(device).backward(retain_graph=True)
            optimizer.step()
        
            train_loss += elbo_loss.item()
            train_reconst_loss += reconst_loss.item()
            train_kld_loss += kld_loss.item()
            train_prop_loss += prop_loss.item()
        reduce_lr_scheduler.step(train_loss)      



def retrain_combined_model(model, input_X):
    """
    Data prep for retraining combined vae model and calling the comb_train function
    
    Parameters
    ----------
    model -> Combined-VAE model
    input_X -> unscaled input data for the Combined-VAE model

    
    Returns
    ---------- 
    None
    """
    input_X = (input_X - input_X.min()) / (input_X.max() - input_X.min())
    input_X = f.pad(input_X, (0, 1), 'constant', 0)
    input_X = input_X.view(input_X.shape[0], 3, 12)
    num_of_dataset = len(input_X)
    num_of_trainset = int(num_of_dataset)
    num_of_testset = 0 #num_of_dataset - num_of_trainset
    dataset = MyDataset(input_X, input_Y)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_0)
    comb_train(model, optimizer, 1, trainloader, loss_coeff=(1, 0.01, 0))



def target_property_wrapper(latent_space, run_id):
    """
    Generates the crystal given the latent variable, evaluate its property, convert it into CIF and save

    Parameters
    ----------
    latent_space -> the candidate latent variable provided by the Bayesian Optimization algorithm
    run_id -> seed of the experiment
    
    Returns
    ---------- 
    pred -> predicted property of generated crystal (formation energy)
    element_z_rs -> elements of the generated crystal
    lattize_z_rs -> lattice of the generated crystal
    coord_z_rs   -> coordinates of the generated crystal
    """
    global lsbo_counter
    if latent_space.shape == (1, combined_z_size):
        latent_space = latent_space.squeeze(0)
    latent_space = torch.Tensor(latent_space).to(torch.float64)#.unsqueeze(0)
    latent_space = latent_space[:combined_z_size].unsqueeze(0)
    with torch.no_grad():
        parts_z = model_combined.decoder.forward(latent_space.to(device))
        element_z_rs, lattize_z_rs, coord_z_rs = rescale_and_divide(parts_z)
        element_z, Y_test_array = get_element_generation(element_z_rs)
        lattize_z = get_lattice_generation(lattize_z_rs, Y_test_array)
        coord_z = get_coor_generation(coord_z_rs, Y_test_array)
        generation_template = X_array_origin[:1, :, :]
        generation_template = torch.Tensor(generation_template).to(device)
        generation_template[:1, 0:103, :4] = torch.Tensor(element_z)[:1, 0:103, :4]
        generation_template[:1, 145:185, :4] = torch.Tensor(element_z)[:1, 103:143, :4]
        x = torch.Tensor(lattize_z[:, 0, :3]) 
        generation_template[:1, 103, :3] = x
        generation_template[:1, 104, :3] = torch.Tensor(lattize_z[:, 1, :3])
        generation_template[:1, 105:145, :4] = torch.Tensor(coord_z)[:1, :40, :]
        ftcp = generation_template[:1, :185, :4]
        ftcp = ftcp.cpu().numpy()
        pred = xgb_model.predict((ftcp.reshape(1, -1)))
        pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, generated_elm_dict = convert_cif(ftcp,
                                        max_elms=max_elms,
                                        max_sites=max_sites,
                                        elm_str=joblib.load('data/element.pkl'),
                                        to_CIF=True,
                                        folder_name='{}/{}/CIF_run_id_{}_{}_EI_bound{}_gamma_{}_roi_var_{}_threshold_{}/'.format(exp_name,exp_name2, run_id, experiment_name, bound, gamma, roi_var, input_threshold),
                                        convertibility_only=False,
                                        print_error=False,
                                        )
        CIF = '{}/{}/CIF_run_id_{}_{}_EI_bound{}_gamma_{}_roi_var_{}_threshold_{}/{}.cif'.format(exp_name,exp_name2, run_id,experiment_name, bound, gamma, roi_var, input_threshold ,lsbo_counter)
        
        if os.path.exists(CIF):
            pred[0] = round(pred[0], 4)
        else:
            print('INVALID')
            pred = np.array([10])

    print('FE prediction is: ',pred)
    if pred < -4:
        file_path = '{}/{}/CIF_run_id_{}_{}_EI_bound{}_gamma_{}_roi_var_{}_threshold_{}/ftcp_{}_pred_{}.npy'.format(exp_name,exp_name2,run_id,experiment_name, bound, gamma, roi_var, input_threshold, lsbo_counter, pred)
        np.save(file_path, ftcp)
    lsbo_counter+=1
    return pred, element_z_rs, lattize_z_rs, coord_z_rs

# load the black box function
xgb_model = xgb.Booster()
with open('{}/xgb_black_box.pkl'.format(exp_name), 'rb') as file:
    xgb_model = pickle.load(file)

# load the gp training dataset
FE_x = torch.load(f'{exp_name}/GP_training_X.pt')
FE_y = torch.load(f'{exp_name}/GP_training_Y.pt')
print(FE_x.shape)
print(FE_y.shape)
FE_x = FE_x.reshape(FE_x.shape[0], 185, -1)
init_element = FE_x[:, 0:143,:]
init_lattice = FE_x[:, 143:145,:3]
init_coor = FE_x[:, 145:185,:]


init_element = torch.tensor(init_element.transpose(0, 2, 1)).double()
init_element = f.pad(init_element[:, :4, :143], (0, 1), 'constant', 0)
_, _, mean_element_init, _, _ = model_element.forward(torch.Tensor(init_element).to(device).to(torch.float64))

init_lattice = torch.tensor(init_lattice.transpose(0, 2, 1)).double()
_, _, mean_lattice_init, _, _ = model_lattice.forward(init_lattice.to(device).to(torch.float64))

init_coor = torch.tensor(init_coor.transpose(0, 2, 1)).double()
_, _, mean_coor_init, _, _ = model_coord.forward(init_coor.to(device).to(torch.float64))
init_set = torch.cat((mean_element_init, mean_lattice_init, mean_coor_init), dim=1)

init_set = (init_set - test_set.min()) / (test_set.max() - test_set.min())
init_set = f.pad(init_set, (0, 1), 'constant', 0)
init_set = init_set.view(init_set.shape[0], 3, 12)
_, _, initial_X, _, _ = model_combined.forward(init_set.to(device))
initial_Y = -FE_y


min_val = np.min(initial_Y)
q1 = np.percentile(initial_Y, 25)  # 1st quartile (25th percentile)
q2 = np.percentile(initial_Y, 50)  # 2nd quartile (50th percentile) or median
q3 = np.percentile(initial_Y, 75)  # 3rd quartile (75th percentile)
max_val = np.max(initial_Y)
threshold = 0
if input_threshold == 0:
    threshold = min_val
elif input_threshold == 1:
    threshold = q1
elif input_threshold == 2:
    threshold = q2
elif input_threshold == 3:
    threshold = q3
elif input_threshold == 4:
    threshold = max_val


lsbo_counter = 0
bounds = torch.tensor([[-bound] * combined_z_size, [bound] * combined_z_size])
run_ids = [0,1,2,3,4,5,6,7,8,9]
for run_id in run_ids:
    objective_values = []
    current_bests = []
    initial_best = -1*np.max(initial_Y, axis=None)

    train_X = torch.tensor(initial_X, dtype=torch.float64).to(device)
    train_Y = torch.tensor(initial_Y, dtype=torch.float64).view(-1, 1).to(device)
    ard_kernel = ScaleKernel(RBFKernel(ard_num_dims=train_X.shape[1]))
    # Create the GP model with ARD kernel
    gp_model = SingleTaskGP(train_X.to(device), train_Y.to(device), likelihood=gpytorch.likelihoods.GaussianLikelihood().to(device))
    gp_model.covar_module = ard_kernel

    mll = ExactMarginalLogLikelihood(gp_model.likelihood.to(device), gp_model.to(device))
    fit_gpytorch_model(mll)
    # Initialize lists to store objective values and current bests
    objective_values = []
    current_bests = []

    # Run optimization manually for each iteration
    max_iter = 1000
    print('bo start')
    torch.manual_seed(run_id)
    np.random.seed(run_id)
    for i in range(max_iter):
        print(i)
        EI = ExpectedImprovement(gp_model, train_Y.max())
        #UCB = UpperConfidenceBound(gp_model, beta=2.0)  # Set beta (kappa in the formula) as desired
        candidate, _ = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
        )
        new_y, element_z_rs, lattize_z_rs, coord_z_rs = target_property_wrapper(candidate.numpy(), run_id)  # Negate as BoTorch maximizes
        new_y = -1 * new_y
        train_X = torch.cat([train_X, candidate.to(device)])
        train_Y = torch.cat([train_Y, torch.tensor([new_y]).to(device)])

        if new_y > (threshold):
            trainloader_element = concat_roi_to_dataset(model_element, element_z_rs, trainloader_element, train_dataset_element)
            trainloader_lattice = concat_roi_to_dataset(model_lattice, lattize_z_rs, trainloader_lattice, train_dataset_lattice)
            trainloader_coord = concat_roi_to_dataset(model_coord, coord_z_rs, trainloader_coord, train_dataset_coord)
            retrain_with_roi(model_element, element_z_rs, 3, trainloader_element, gamma)
            retrain_with_roi(model_lattice, lattize_z_rs, 3, trainloader_lattice, gamma)
            retrain_with_roi(model_coord, coord_z_rs, 3, trainloader_coord, gamma)
            mean_element = get_updated_embeddings(model_element, trainloader_element)
            mean_lattice = get_updated_embeddings(model_lattice, trainloader_lattice)
            mean_coord = get_updated_embeddings(model_coord, trainloader_coord)
            comb_train_set = torch.cat((mean_element, mean_lattice, mean_coord), dim=1)
            retrain_combined_model(model_combined, comb_train_set)
            print("***********************")
            print("***********************")
            print("***********************")
            print("********** ALL RT ENDED *************")

        gp_model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        fit_gpytorch_model(mll)


        df = pd.DataFrame(train_Y.detach().cpu().numpy())
        df.to_csv('{}/{}/lsbo_feb28_run_{}_LD_{}_bound_{}_EI_gamma_{}_roi_var_{}_threshold_{}.csv'.format(exp_name,exp_name2, run_id, combined_z_size, bound, gamma, roi_var, input_threshold), index=False)
    
    # after each seed, delete the retrained models and load the pre-trained versions
    del model_coord, model_element, model_lattice, model_combined
    model_combined = VAE(combined_z_size, input_X_comb, input_Y).to(device).to(torch.float64)
    model_combined.load_state_dict(torch.load('{}/{}'.format(exp_name,ckpt_name), map_location=torch.device(device)))
    model_coord = VAE(coord_z_size, input_X_coord, input_Y).to(device).to(torch.float64)
    model_coord.load_state_dict(torch.load('{}/coordinate_vae.pt'.format(exp_name), map_location=torch.device(device)))
    model_lattice = VAE_Lattice(lattice_z_size, input_X_lattice, input_Y).to(device).to(torch.float64)
    model_lattice.load_state_dict(torch.load('{}/lattice_vae.pt'.format(exp_name), map_location=torch.device(device)))
    model_element = VAE(element_z_size, input_X_element, input_Y).to(device).to(torch.float64)
    model_element.load_state_dict(torch.load(f'crystal_lsbo_repo/element_vae.pt', map_location=torch.device(device)))
    

    dataset_element = MyDataset(input_X_element, input_Y)
    train_dataset_element, test_dataset_element = random_split(dataset_element, [num_of_trainset, num_of_testset], generator=generator1)
    trainloader_element = DataLoader(train_dataset_element, batch_size=batch_size, shuffle=True)

    dataset_coord = MyDataset(input_X_coord, input_Y)
    train_dataset_coord, test_dataset_coord = random_split(dataset_coord, [num_of_trainset, num_of_testset],generator=generator1)
    trainloader_coord = DataLoader(train_dataset_coord, batch_size=batch_size, shuffle=True)

    dataset_lattice = MyDataset(input_X_lattice, input_Y)
    train_dataset_lattice, test_dataset_lattice = random_split(dataset_lattice, [num_of_trainset, num_of_testset],generator=generator1)
    trainloader_lattice = DataLoader(train_dataset_lattice, batch_size=batch_size, shuffle=True)










