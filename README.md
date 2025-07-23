# Crystal-LSBO

Check out our paper here: [Crystal-LSBO Paper](https://direct.mit.edu/neco/article/37/8/1505/131386/Crystal-LSBO-Automated-Design-of-De-Novo-Crystals)

Welcome to Crystal-LSBO!

Contents:
- crystal_lsbo_repo -> includes trained models, and necessary files for running experiments
- data -> part of the required data, the rest will be downloaded and prepared while running the code
- legacy_version -> includes single .py files for random and lca-lsbo generation codes. move files to the main directory to use 
these versions
    - crystal-lca-lsbo.py -> The code for Crystal-LCA-LSBO experiments provided in the paper
    - random_gen_check_validity.py -> The code for random generations with standard normal distribution and checking validities
- data_utils.py -> functions for data preparation
- dataset.py -> custom torch dataset
- random_experiment.py -> code for random generation
- run_random.py -> wrapper for random generation
- run_lca_lsbo.py -> code for lca-lsbo
- vis_umap.py -> The code to create UMAP plot in Figure 1(B).

First, please install the necessary libraries using

> pip install -r requirements.txt

Next, to reproduce the validity results from randomly generated crystals from standard normal distribution, please run the below

> python run_random.py 20 combined_vae.pt

Replace run_random.py with random_gen_check_validity.py to run the legacy version.

Note that you will need to create a Materials Project API key and copy it within the code in order to run. This key is required to get the crystal data described in the paper. (https://next-gen.materialsproject.org/api)

After crystal generation is completed, you can reproduce the Figure 1(B) in the paper using below command

> python vis_umap.py

For the LSBO experiments, you can reproduce the Crystal-LCA-LSBO experiment results using below code

> python run_lca_lsbo.py 20 combined_vae.pt 3 "$gamma" "$roi_var" "$threshold"

if we are to set gamma = 2.5, roi_var = 0.3, threshold = 4,

> python run_lca_lsbo.py 20 combined_vae.pt 3 2.5 0.3 4

Replace run_lca_lsbo.py with crystal-lca-lsbo.py to run the legacy version.
