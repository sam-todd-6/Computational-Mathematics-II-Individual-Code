import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements

L = 200              # system size (L x L)
M = 500              # number of realisations
p_c = 0.5927          # reference probability (~pc for 2D site percolation)
g_values = [0.02, 0.04, 0.06, 0.08, 0.10]  # grad strengths

np.random.seed(1) # Added right at the end - after percolation graphs created - for reproducibility

def front_position_and_width(labels):

    # labels touching top and bottom boundries
    top_labels = np.unique(labels[0, labels[0] > 0])
    bottom_labels = np.unique(labels[-1, labels[-1] > 0])

    # vertically spanning clusters
    spanning_labels = np.intersect1d(top_labels, bottom_labels)

    if len(spanning_labels) == 0: #error avoidance
        return np.nan, np.nan

    # chosen largest spanning cluster
    sizes = [np.sum(labels == lab) for lab in spanning_labels]
    span_label = spanning_labels[np.argmax(sizes)]

    #y-coordinatess of all sites in the spaning cluster
    y_coords = np.where(labels == span_label)[0]

    front_mean = np.mean(y_coords) #find stats to report
    front_width = np.std(y_coords)

    return front_mean, front_width


mean_front_pos = []
mean_front_wid = []

for g in g_values:

    front_means = []
    front_widths = []

    for _ in range(M):

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

        # cluster labelling
        labels, _ = measurements.label(occ)

        # front statistics
        fm, fw = front_position_and_width(labels)

        if not np.isnan(fm):
            front_means.append(fm)
            front_widths.append(fw)

    # averages for this g
    mean_front_pos.append(np.mean(front_means))
    mean_front_wid.append(np.mean(front_widths))


plt.figure(figsize=(8.7, 5.5))
plt.plot(g_values, mean_front_wid, 'o-', lw=3, markersize=8)
plt.xlabel('Gradient strength $g$', fontsize=28)
plt.ylabel('Mean front width', fontsize=28)
plt.title('Percolation front width vs gradient strength', fontsize=30)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.grid(True)
plt.tight_layout()
plt.show()
