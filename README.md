# Crystal-LSBO

Welcome to Crystal-LSBO <https://arxiv.org/abs/2405.17881>

Contents:
- crystal_lsbo_repo -> includes trained models, and necessary files for running experiments
- data -> part of the required data, the rest will be downloaded and prepared while running the code
- crystal-lca-lsbo.py -> The code for Crystal-LCA-LSBO experiments provided in the paper
- random_gen_check_validity.py -> The code for random generations with standard normal distribution and checking validities
- vis_umap.py -> The code to create UMAP plot in Figure 1(B).

First, please install the necessary libraries using

> pip install -r requirements.txt

Next, to reproduce the validity results from randomly generated crystals from standard normal distribution, please run the below

> python random_gen_check_validity.py 20 combined_vae.pt

Note that you will need to create a Materials Project API key and copy it within the code in order to run. This key is required to get the crystal data described in the paper. (https://next-gen.materialsproject.org/api)

After crystal generation is completed, you can reproduce the Figure 1(B) in the paper using below command

> python vis_umap.py

For the LSBO experiments, you can reproduce the Crystal-LCA-LSBO experiment results using below code

> python crystal-lca-lsbo.py 20 combined_vae.pt 3 "$gamma" "$roi_var" "$threshold"

if we are to set gamma = 2.5, roi_var = 0.3, threshold = 4,

> python crystal-lca-lsbo.py 20 combined_vae.pt 3 2.5 0.3 4
