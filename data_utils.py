# data_utils.py
import os
import pickle
import math
import numpy as np
import pandas as pd
import json
import joblib
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from ase.io import write
from ase import spacegroup


random_counter = 0

def minmax_(X_array, Y_array, scaler_path=None):
    """
    Applies minmax scaling.
    """
    dim0, dim1, dim2 = X_array.shape
    scaler_x = MinMaxScaler()
    temp_ = np.transpose(X_array, (1, 0, 2)).reshape(dim1, dim0*dim2)
    temp_ = scaler_x.fit_transform(temp_.T).T.reshape(dim1, dim0, dim2)
    X_normed = np.transpose(temp_, (1, 0, 2))
    
    scaler_y = MinMaxScaler()
    Y_normed = scaler_y.fit_transform(Y_array) 
    
    if scaler_path is not None:
        with open(scaler_path + "_scaler_X.pkl", "wb") as f:
            pickle.dump(scaler_x, f)
        with open(scaler_path + "_scaler_Y.pkl", "wb") as f:
            pickle.dump(scaler_y, f)
    return X_normed, Y_normed, scaler_x, scaler_y

def inv_minmax(X_normed, Y_normed, scaler_x, scaler_y):
    """
    Inverse minmax scaling.
    """
    dim0, dim1, dim2 = X_normed.shape
    temp_ = X_normed.reshape(dim0*dim1, dim2)
    temp_ = scaler_x.inverse_transform(temp_)
    X = temp_.reshape(dim0, dim1, dim2).transpose(0, 2, 1)
    Y = scaler_y.inverse_transform(Y_normed)
    return X, Y

def data_query(mp_api_key, max_elms=3, min_elms=3, max_sites=20, include_te=False):
    """
    Query crystal data from the Materials Project.
    """
    mpdr = MPRester(mp_api_key)
    query_criteria = {
        'e_above_hull': {'$lte': 0.08},
        'nelements': {'$gte': min_elms, '$lte': max_elms},
        'nsites': {'$lte': max_sites},
    }
    query_properties = [
        'material_id',
        'formation_energy_per_atom',
        'band_gap',
        'pretty_formula',
        'e_above_hull',
        'elements',
        'cif',
        'spacegroup.number'
    ]
    materials = mpdr.query(criteria=query_criteria, properties=query_properties)
    dataframe = pd.DataFrame([i for i in materials])
    dataframe['ind'] = np.arange(len(dataframe))
    
    if include_te:
        dataframe['ind'] = np.arange(len(dataframe))
        te = pd.read_csv('data/thermoelectric_prop.csv', index_col=0).dropna()
        ind = dataframe.index.intersection(te.index)
        dataframe = pd.concat([dataframe, te.loc[ind,:]], axis=1)
        dataframe['Seebeck'] = dataframe['Seebeck'].apply(np.abs)
    
    return dataframe

def FTCP_represent(dataframe, max_elms=3, max_sites=20, return_Nsites=False):
    """
    Converts CIFs into FTCP representations.
    """
    import warnings
    warnings.filterwarnings("ignore")
    
    elm_str = joblib.load('data/element.pkl')
    elm_onehot = OneHotEncoder().fit_transform(np.arange(1, len(elm_str)+1)[:, None]).toarray()
    
    with open('data/atom_init.json') as f:
        elm_prop = json.load(f)
    elm_prop = {int(key): value for key, value in elm_prop.items()}
    
    FTCP = []
    if return_Nsites:
        Nsites = []
    op = tqdm(dataframe.index)
    for idx in op:
        op.set_description('representing data as FTCP ...')
        crystal = Structure.from_str(dataframe['cif'][idx], fmt="cif")
        elm, elm_idx = np.unique(crystal.atomic_numbers, return_index=True)
        site_elm = np.array(crystal.atomic_numbers)
        elm = site_elm[np.sort(elm_idx)]
        ELM = np.zeros((len(elm_onehot), max(max_elms, 3)))
        ELM[:, :len(elm)] = elm_onehot[elm-1, :].T
        latt = crystal.lattice
        LATT = np.array((latt.abc, latt.angles))
        LATT = np.pad(LATT, ((0, 0), (0, max(max_elms, 3)-LATT.shape[1])), constant_values=0)
        SITE_COOR = np.array([site.frac_coords for site in crystal])
        SITE_COOR = np.pad(SITE_COOR, ((0, max_sites-SITE_COOR.shape[0]), (0, max(max_elms, 3)-SITE_COOR.shape[1])), constant_values=0)
        elm_inverse = np.zeros(len(crystal), dtype=int)
        for count, e in enumerate(elm):
            elm_inverse[np.argwhere(site_elm == e)] = count
        SITE_OCCU = OneHotEncoder().fit_transform(elm_inverse[:, None]).toarray()
        SITE_OCCU = np.pad(SITE_OCCU, ((0, max_sites-SITE_OCCU.shape[0]), (0, max(max_elms, 3)-SITE_OCCU.shape[1])), constant_values=0)
        ELM_PROP = np.zeros((len(elm_prop[1]), max(max_elms, 3)))
        ELM_PROP[:, :len(elm)] = np.array([elm_prop[e] for e in elm]).T
        REAL = np.concatenate((ELM, LATT, SITE_COOR, SITE_OCCU, np.zeros((1, max(max_elms, 3))), ELM_PROP), axis=0)
        recip_latt = latt.reciprocal_lattice_crystallographic
        hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.297, zip_results=False)
        if len(hkl) < 60:
            hkl, g_hkl, ind, _ = recip_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], 1.4, zip_results=False)
        not_zero = g_hkl != 0
        hkl = hkl[not_zero, :].astype('int16')
        hkl_sum = np.sum(np.abs(hkl), axis=1)
        h = -hkl[:, 0]
        k = -hkl[:, 1]
        l = -hkl[:, 2]
        hkl_idx = np.lexsort((l, k, h, hkl_sum))[:59]
        hkl = hkl[hkl_idx, :]
        g_hkl = g_hkl[hkl_idx]
        k_dot_r = np.einsum('ij,kj->ik', hkl, SITE_COOR[:, :3])
        F_hkl = np.matmul(np.pad(ELM_PROP[:, elm_inverse], ((0, 0), (0, max_sites-len(elm_inverse))), constant_values=0), np.pi * k_dot_r.T)
        RECIP = np.zeros((REAL.shape[0], 59))
        RECIP[-ELM_PROP.shape[0]-1, :] = g_hkl
        RECIP[-ELM_PROP.shape[0]:, :] = F_hkl
        FTCP.append(np.concatenate([REAL, RECIP], axis=1))
        if return_Nsites:
            Nsites.append(len(crystal))
    FTCP = np.stack(FTCP)
    if not return_Nsites:
        return FTCP
    else:
        return FTCP, np.array(Nsites)

def pad(FTCP, pad_width):
    """
    Zero-pads the FTCP representation along the second dimension.
    """
    return np.pad(FTCP, ((0, 0), (0, pad_width), (0, 0)), constant_values=0)

def convert_cif(gen_structures,
                max_elms=4,
                max_sites=40,
                elm_str=None,
                one_hot_threshold=0.01,
                site_occu_threshold=0.01,
                to_CIF=True,
                folder_name=None,
                convertibility_only=False,
                print_error=False,
                oxygen_distance_distribution=False):
    """
    Converts generated structure representations into CIF files.
    """
    from ase import spacegroup
    from ase.io import write

    global random_counter
    if elm_str is None:
        elm_str = joblib.load('data/element.pkl')
    Ntotal_elms = len(elm_str)
    generated_elm_dict = {elm: 0 for elm in elm_str}
    one_hot_part = np.pad(gen_structures[:, :Ntotal_elms, :max_elms],
                           ((0, 0), (0, 1), (0, 0)),
                           constant_values=one_hot_threshold)
    pred_elm = np.argmax(one_hot_part, axis=1)
    
    pred_formula = []
    coordinate_temp = []
    site_occu_part = gen_structures[:, Ntotal_elms+2+max_sites:Ntotal_elms+2+2*max_sites, :max_elms]
    site_occu_part = np.pad(site_occu_part, ((0, 0), (0, 0), (0, 1)), constant_values=site_occu_threshold)
    max_site_occu_indices = np.argmax(site_occu_part, axis=2)
    for i in range(site_occu_part.shape[0]):
        temp = []
        for j in range(site_occu_part.shape[1]):
            if max_site_occu_indices[i][j] != max_elms and pred_elm[i][max_site_occu_indices[i][j]] != Ntotal_elms:
                temp.append([j, pred_elm[i][max_site_occu_indices[i][j]]])
        if len(temp) == 0:
            pred_formula.append([])
            coordinate_temp.append([])
        else:
            pred_formula.append([elm_str[int(t[1])] for t in temp])
            coordinate_temp.append([int(t[0]) for t in temp])
        for e in pred_formula[-1]:
            generated_elm_dict[e] += 1
    pred_abc = gen_structures[:, Ntotal_elms, :3]
    pred_ang = gen_structures[:, Ntotal_elms+1, :3]
    pred_latt = np.concatenate((pred_abc, pred_ang), axis=1)
    
    pred_site_coor = []
    pred_site_coor_ = gen_structures[:, Ntotal_elms+2:Ntotal_elms+2+max_sites, :3]
    for i, c in enumerate(pred_formula):
        Nsites = len(c)
        if Nsites == 0:
            pred_site_coor.append([])
        else:   
            pred_site_coor.append(pred_site_coor_[i, coordinate_temp[i], :])
    
    assert (len(pred_formula) == len(pred_site_coor) and len(pred_formula) == len(pred_latt))
    if convertibility_only:
        available_count = 0
        for j in range(len(pred_formula)):
            if len(pred_formula[j]) == 0:
                if print_error:
                    print("Could not write the file, all numbers in generated matrix of elements are too low.")
                continue
            try:
                crystal = spacegroup.crystal(pred_formula[j],
                                             basis=(pred_site_coor[j]),
                                             cellpar=(pred_latt[j]))
                available_count += 1
            except Exception as e:
                if print_error:
                    print(f"Error: {e}")
        return available_count, generated_elm_dict
    if oxygen_distance_distribution:
        nearest_elements_list = []
        nearest_elements_distance_list = []
        for j in range(len(pred_formula)):
            if len(pred_formula[j]) == 0:
                if print_error:
                    print("Could not write the file, all numbers in generated matrix of elements are too low.")
                continue
            try:
                crystal = spacegroup.crystal(pred_formula[j],
                                             basis=(pred_site_coor[j]),
                                             cellpar=(pred_latt[j]))
                distances_matrix = crystal.get_all_distances(mic=False)
                elements_list = crystal.get_chemical_symbols()
                for i_elem, elem in enumerate(elements_list):
                    if str(elem) == 'O':
                        second_smallest_index = np.argsort(distances_matrix[i_elem])[1]
                        nearest_elements_list.append(elements_list[second_smallest_index])
                        nearest_elements_distance_list.append(distances_matrix[i_elem][second_smallest_index])
            except Exception as e:
                if print_error:
                    print(f"Error: {e}")
        return nearest_elements_list, nearest_elements_distance_list
    if to_CIF:
        os.makedirs(folder_name, exist_ok=True)
        global random_counter
        for j in range(len(pred_formula)):
            try:
                crystal = spacegroup.crystal(pred_formula[j],
                                             basis=(pred_site_coor[j]),
                                             cellpar=(pred_latt[j]))
                crystal_save_path = os.path.join(folder_name, str(random_counter)+'.cif')
                write(crystal_save_path, crystal)
            except Exception as e:
                print(f"Could not write the file: {e}")
    random_counter += 1
    return pred_formula, pred_abc, pred_ang, pred_latt, pred_site_coor, generated_elm_dict

