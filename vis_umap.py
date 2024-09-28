import torch
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

exp_name = 'crystal_lsbo_repo/'
dir_name = 'crystal_random'
run_id = 0
# Load concatenated candidates
concatenated_candidates = torch.load(f'{exp_name}/{dir_name}/concatenated_candidates_seed_{run_id}.pt').numpy()

# Load labels
labels_df = pd.read_csv(f'{exp_name}/{dir_name}/labels_seed_{run_id}.csv')
labels = labels_df.values.squeeze()  # Assuming labels are in a single column
color_labels = ['red' if label == -1 else 'blue' for label in labels]


umap_reducer = umap.UMAP(n_components=2, random_state = 8) # 8 is selected because of the overlapped regions between legend and the scatter points
candidates_umap = umap_reducer.fit_transform(concatenated_candidates)

# Plotting
plt.figure(figsize=(12, 12))
for color in set(color_labels):
    label = 'Invalid' if color == 'red' else 'Valid'
    condition = [c == color for c in color_labels]
    plt.scatter(candidates_umap[condition, 0], candidates_umap[condition, 1], c=color, label=label, s=170, alpha=0.6)
plt.xlabel('UMAP Dimension 1', fontsize = 55)
plt.ylabel('UMAP Dimension 2', fontsize = 55)
plt.title('B', fontsize=60)
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', handlelength=0.3, handletextpad=0.2, fontsize = 40,  bbox_to_anchor=(1.025,1.025))
plt.savefig(f'{exp_name}/{dir_name}/umap_crystal-lsbo.pdf', bbox_inches="tight")
