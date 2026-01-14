import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements


L = 200                  # lattice size
M = 500                  # number of realisations
p_c = 0.5927             # 2D site percolation threshold
g = 0.2                  # total gradient across system
tol = 0.01               # width of critical strip in p

all_cluster_sizes = []

for i in range(M):

    # make random field
    grid = np.random.rand(L, L)
    
    # vertical coordinate (0 at top, 1 at bottom)
    y = np.linspace(0, 1, L)
    
    # probability function
    a = p_c
    p_y = a + g * (y - 0.5)
    
    # build occupation grid
    occ = np.zeros((L, L), dtype=bool)
    for j in range(L):
        occ[j, :] = grid[j, :] < p_y[j] # vectorised code to find which squares should be coloured

    ##### New below - previous is same as Figure 1 #####  

    # label clusters
    labels, num = measurements.label(occ)

    # find rows where p(y) ≈ p_c
    strip_rows = np.where(np.abs(p_y - p_c) < tol)[0]

    # mask only the critical strip
    strip_mask = np.zeros_like(labels, dtype=bool)
    strip_mask[strip_rows, :] = True

    # apply mask
    strip_labels = labels * strip_mask

    # remove zeros (background)
    strip_labels = strip_labels[strip_labels > 0]

    # count cluster sizes in strip
    unique, counts = np.unique(strip_labels, return_counts=True)

    # store sizes
    all_cluster_sizes.extend(counts)

# convert to array
all_cluster_sizes = np.array(all_cluster_sizes)


bins = np.logspace(np.log10(all_cluster_sizes.min()), np.log10(all_cluster_sizes.max()), 25)
hist, bin_edges = np.histogram(all_cluster_sizes, bins=bins)

# bin centres
bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])


apx=11
plt.figure(figsize=(apx, apx/8.7*5.2))
plt.loglog(bin_centres, hist, 'o', markersize=8)
plt.xlabel('Cluster size $s$', fontsize=28)
plt.ylabel('Count', fontsize=28)
plt.title('Cluster size distribution near critical strip', fontsize=30)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.show()
