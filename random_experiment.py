# experiment.py
import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from torch.utils.data import DataLoader, random_split

from models import VAE, VAE_Lattice
from dataset import MyDataset
from data_utils import minmax_, inv_minmax, data_query, FTCP_represent, pad, convert_cif

def run_experiment(combined_z_size, ckpt_name):
    # ------------------- Parameters ----------------------
    experiment_name = 'crystal_random'
    exp_name = "crystal_lsbo_repo"
    exp_name2 = "crystal_random"
    element_z_size = 16
    coord_z_size = 16
    max_elms = 4
    min_elms = 3
    max_sites = 40
    mp_api_key = ''  # <<< INSERT YOUR MATERIALS PROJECT API KEY HERE
    epochs = 500
    lr_0 = 5e-4
    batch_size = 256
    loss_coeff = (5, 10, 1)
    property_for_pridict = ['formation_energy_per_atom', 'band_gap']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    # -----------------------------------------------------

    # --- Load or query the data ---
    data_csv_path = f"{exp_name}/data.csv"
    if os.path.exists(data_csv_path):
        dataframe = pd.read_csv(data_csv_path)
    else:
        dataframe = data_query(mp_api_key, max_elms, min_elms, max_sites)
        os.makedirs(exp_name, exist_ok=True)
        dataframe.to_csv(data_csv_path, index=False)
    
    ftcp_path = f'{exp_name}/FTCP_representation.pkl'
    nsites_path = f'{exp_name}/Nsites.pkl'
    if os.path.exists(ftcp_path) and os.path.exists(nsites_path):
        with open(ftcp_path, 'rb') as file:
            FTCP_representation = pickle.load(file)
        with open(nsites_path, 'rb') as file:
            Nsites = pickle.load(file)
    else:
        FTCP_representation, Nsites = FTCP_represent(dataframe, max_elms, max_sites, return_Nsites=True)
        FTCP_representation = pad(FTCP_representation, 2)
        os.makedirs(exp_name, exist_ok=True)
        with open(ftcp_path, 'wb') as file:
            pickle.dump(FTCP_representation, file)
        with open(nsites_path, 'wb') as file:
            pickle.dump(Nsites, file)
    FTCP_representation = pad(FTCP_representation, 2)
    print("FTCP_representation:", FTCP_representation.shape)
    
    X_array_origin = FTCP_representation
    del FTCP_representation
    Y_array = dataframe[property_for_pridict].values
    del dataframe
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # --- Create subsets for the different models ---
    subset_1 = X_array_origin[:, 0:103, :4]  # element
    subset_2 = X_array_origin[:, 103:104, :3]  # angle
    subset_3 = X_array_origin[:, 104:105, :3]  # cell lengths
    subset_4 = X_array_origin[:, 105:145, :4]  # coordinates
    subset_5 = X_array_origin[:, 145:185, :4]  # occupancy
    
    # --------- Load Element Model ---------
    X_array = np.concatenate((subset_1, subset_5), axis=1)
    del subset_1, subset_5
    input_normed, Y_normed, scaler_x_element, scaler_y_element = minmax_(X_array, Y_array)
    input_X_full = torch.tensor(input_normed.transpose(0, 2, 1)).double()
    input_Y = torch.tensor(Y_normed).double()
    input_X_element = F.pad(input_X_full[:, :4, :143], (0, 1), 'constant', 0)
    num_of_dataset = len(input_X_element)
    num_of_trainset = 40000
    num_of_testset = num_of_dataset - num_of_trainset
    dataset_element = MyDataset(input_X_element, input_Y)
    generator1 = torch.Generator().manual_seed(42)
    train_dataset_element, test_dataset_element = random_split(dataset_element, [num_of_trainset, num_of_testset], generator=generator1)
    trainloader_element = DataLoader(train_dataset_element, batch_size=batch_size, shuffle=True)
    testloader_element = DataLoader(test_dataset_element, batch_size=batch_size, shuffle=True)
    model_element = VAE(element_z_size, input_X_element, input_Y).to(device).to(torch.float64)
    model_element.load_state_dict(torch.load(f'{exp_name}/element_vae.pt', map_location=device))
    _, _, mean_element_train, _, _ = model_element.forward(torch.stack([dataset_element[i][0] for i in train_dataset_element.indices]).to(device))
    _, _, mean_element_test, _, _ = model_element.forward(torch.stack([dataset_element[i][0] for i in test_dataset_element.indices]).to(device))
    print("Element model latent shape (train):", mean_element_train.shape)
    print("Element model latent shape (test):", mean_element_test.shape)
    
    # --------- Load Lattice Model ---------
    lattice_z_size = 3
    X_array = np.concatenate((subset_2, subset_3), axis=1)
    del subset_2, subset_3
    X_array = X_array[:, :, :4]
    input_normed, Y_normed, scaler_x_lattice, scaler_y_lattice = minmax_(X_array, Y_array)
    input_X_full = torch.tensor(input_normed.transpose(0, 2, 1)).double()
    input_Y = torch.tensor(Y_normed).double()
    input_X_lattice = input_X_full[:, :4, :]
    dataset_lattice = MyDataset(input_X_lattice, input_Y)
    train_dataset_lattice, test_dataset_lattice = random_split(dataset_lattice, [num_of_trainset, num_of_testset], generator=generator1)
    trainloader_lattice = DataLoader(train_dataset_lattice, batch_size=batch_size, shuffle=True)
    testloader_lattice = DataLoader(test_dataset_lattice, batch_size=batch_size, shuffle=True)
    model_lattice = VAE_Lattice(lattice_z_size, input_X_lattice, input_Y).to(device).to(torch.float64)
    model_lattice.load_state_dict(torch.load(f'{exp_name}/lattice_vae.pt', map_location=device))
    _, _, mean_lattice_train, _, _ = model_lattice.forward(torch.stack([dataset_lattice[i][0] for i in train_dataset_lattice.indices]).to(device))
    _, _, mean_lattice_test, _, _ = model_lattice.forward(torch.stack([dataset_lattice[i][0] for i in test_dataset_lattice.indices]).to(device))
    
    # --------- Load Coordinate Model ---------
    X_array = subset_4
    del subset_4
    input_normed, Y_normed, scaler_x_coor, scaler_y_coor = minmax_(X_array, Y_array)
    input_X_full = torch.tensor(input_normed.transpose(0, 2, 1)).double()
    input_Y = torch.tensor(Y_normed).double()
    input_X_coord = input_X_full[:, :4, :]
    dataset_coord = MyDataset(input_X_coord, input_Y)
    train_dataset_coord, test_dataset_coord = random_split(dataset_coord, [num_of_trainset, num_of_testset], generator=generator1)
    trainloader_coord = DataLoader(train_dataset_coord, batch_size=batch_size, shuffle=True)
    testloader_coord = DataLoader(test_dataset_coord, batch_size=batch_size, shuffle=True)
    model_coord = VAE(coord_z_size, input_X_coord, input_Y).to(device).to(torch.float64)
    model_coord.load_state_dict(torch.load(f'{exp_name}/coordinate_vae.pt', map_location=device))
    _, _, mean_coord_train, _, _ = model_coord.forward(torch.stack([dataset_coord[i][0] for i in train_dataset_coord.indices]).to(device))
    _, _, mean_coord_test, _, _ = model_coord.forward(torch.stack([dataset_coord[i][0] for i in test_dataset_coord.indices]).to(device))
    
    # Combine latent features (not used further here, but available for analysis)
    train_set = torch.cat((mean_element_train, mean_lattice_train, mean_coord_train), dim=1)
    test_set = torch.cat((mean_element_test, mean_lattice_test, mean_coord_test), dim=1)
    
    # --------- Load Combined Model ---------
    input_X_comb = torch.load(f'{exp_name}/combined_vae_input_latent_data.pt')
    data_min = input_X_comb.min()
    data_max = input_X_comb.max()
    input_X_comb = F.pad(input_X_comb, (0, 1), 'constant', 0)
    input_X_comb = input_X_comb.view(input_X_comb.shape[0], 3, 12)
    dataset_combined = MyDataset(input_X_comb, input_Y)
    train_dataset_comb, test_dataset_comb = random_split(dataset_combined, [1, len(input_X_comb)-1], generator=generator1)
    trainloader_comb = DataLoader(train_dataset_comb, batch_size=batch_size, shuffle=True)
    testloader_comb = DataLoader(test_dataset_comb, batch_size=batch_size, shuffle=True)
    model_combined = VAE(combined_z_size, input_X_comb, input_Y).to(device).to(torch.float64)
    model_combined.load_state_dict(torch.load(f'{exp_name}/{ckpt_name}', map_location=device))
    
    # ------------- Helper Functions -----------------
    def rescale_and_divide(x_recon):
        x_recon = x_recon.view(-1, 36)
        x_recon = x_recon * (data_max - data_min) + data_min
        element_z = x_recon[:, :element_z_size]
        lattize_z = x_recon[:, element_z_size:(element_z_size+3)]
        coord_z = x_recon[:, (element_z_size+3):(element_z_size+3+coord_z_size)]
        return element_z, lattize_z, coord_z

    def get_lattice_generation(lattize_z_sample_first, Y_test_array):
        lattize_z_sample = model_lattice.decoder.forward(lattize_z_sample_first.to(device).to(torch.float64))
        lattize_z_sample = torch.Tensor(lattize_z_sample).to(device)
        lattize_z_sample = lattize_z_sample.detach().cpu().numpy()
        lattize_z_sample, _ = inv_minmax(lattize_z_sample, Y_test_array, scaler_x_lattice, scaler_y_lattice)
        return lattize_z_sample

    def get_coor_generation(coord_z_sample_first, Y_test_array):
        coord_z_sample = model_coord.decoder.forward(coord_z_sample_first.to(device).to(torch.float64))
        coord_z_sample = torch.Tensor(coord_z_sample).to(device)
        coord_z_sample = coord_z_sample.detach().cpu().numpy()
        coord_z_sample, _ = inv_minmax(coord_z_sample, Y_test_array, scaler_x_coor, scaler_y_coor)
        return coord_z_sample

    # For element generation, we re-use the element model’s scalers:
    input_normed, Y_normed, scaler_x, scaler_y = minmax_(np.concatenate((X_array_origin[:, 0:103, :4],
                                                                             X_array_origin[:, 145:185, :4]), axis=1), Y_array)
    input_X = torch.tensor(input_normed.transpose(0, 2, 1)).double()
    input_Y = torch.tensor(Y_normed).double()
    num_of_dataset = len(input_X)
    indices = torch.randperm(num_of_dataset)[:1000]
    X_test = input_X[indices].to(device)
    Y_test = input_Y[indices].to(device)
    
    def get_element_generation(element_z_sample_first):
        with torch.no_grad():
            element_z_gen = model_element.decoder.forward(element_z_sample_first.to(device).to(torch.float64))
        Y_test_array = Y_test.detach().cpu().numpy()
        element_z_gen = element_z_gen.detach().cpu().numpy()
        X_test_gen_array = np.zeros((1, X_test.shape[1], X_test.shape[2]))
        X_test_gen_array[:, :4, 0:143] = element_z_gen[:, :4, 0:143]
        X_test_gen_array, _ = inv_minmax(X_test_gen_array, Y_test_array, scaler_x, scaler_y)
        return X_test_gen_array, Y_test_array

    global random_counter
    random_counter = 0
    def target_property_wrapper(latent_space, run_id):
        global random_counter
        if latent_space.shape == (1, combined_z_size):
            latent_space = latent_space.squeeze(0)
        latent_space = torch.Tensor(latent_space).to(torch.float64)
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
            cif_folder = f'{exp_name}/{exp_name2}/CIF_run_id_{run_id}_{experiment_name}'
            pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, generated_elm_dict = convert_cif(
                ftcp,
                max_elms=max_elms,
                max_sites=max_sites,
                elm_str=joblib.load('data/element.pkl'),
                to_CIF=True,
                folder_name=cif_folder,
                convertibility_only=False,
                print_error=False,
            )
            CIF = f'{cif_folder}/{random_counter}.cif'
            if os.path.exists(CIF):
                pred = np.array([1])
            else:
                pred = np.array([-1])
        random_counter += 1
        return pred, element_z_rs, lattize_z_rs, coord_z_rs

    # Load the black-box model (e.g. from XGBoost)
    xgb_model = xgb.Booster()
    with open(f'{exp_name}/xgb_black_box.pkl', 'rb') as file:
        xgb_model = pickle.load(file)
    
    run_ids = [0,1,2,3,4,5,6,7,8,9]
    for run_id in run_ids:
        max_iter = 1000
        torch.manual_seed(run_id)
        np.random.seed(run_id)
        candidates_list = []  
        res_tensor = torch.tensor([], device=device)
        for i in range(max_iter):
            print(i)
            candidate = torch.randn([1, combined_z_size])
            new_y, element_z_rs, lattize_z_rs, coord_z_rs = target_property_wrapper(candidate.numpy(), run_id)
            candidates_list.append(candidate.detach())
            res_tensor = torch.cat([res_tensor, torch.tensor([new_y]).to(device)])
        df = pd.DataFrame(res_tensor.detach().cpu().numpy())
        os.makedirs(f'{exp_name}/{exp_name2}', exist_ok=True)
        df.to_csv(f'{exp_name}/{exp_name2}/labels_seed_{run_id}.csv', index=False)
        concatenated_candidates = torch.cat(candidates_list, dim=0)
        torch.save(concatenated_candidates, f'{exp_name}/{exp_name2}/concatenated_candidates_seed_{run_id}.pt')

if __name__ == "__main__":
    # Allow running experiment.py directly for testing purposes.
    import sys
    if len(sys.argv) >= 3:
        combined_z_size = int(sys.argv[1])
        ckpt_name = sys.argv[2]
        run_experiment(combined_z_size, ckpt_name)
    else:
        print("Please provide combined_z_size and ckpt_name as command line arguments.")

