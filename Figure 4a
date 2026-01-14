import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

seed=7 # [3, 3, 7, 9]
gridy = 100
gridx = 50
gravityvalue = 2 # [0, 0.6, 2, 10]

def neighbouringsites(row, column, maxgridy, maxgridx): # maxgrid vars are maximum valid values 
                                                        # potentially confusing since 0 indexed
    retlist = []                                        # so maxgridy = gridy - 1
    if 0 < row:
        retlist.append((row-1, column)) # up
    if 0 < column:
        retlist.append((row, column-1)) # left
    if row < maxgridy:
        retlist.append((row+1, column)) # down
    if column < maxgridx:
        retlist.append((row, column+1)) # right
    
    return retlist


rng = np.random.RandomState(seed)
thresholds = rng.rand(gridy, gridx)
invaded = np.zeros((gridy, gridx))

# Initial invasion: top row invaded (or single center cell if desired) 

'''FOR NOW ALL TOP IS PERCED'''
percfront = set()

invaded[0, :] = 1 # 1 IS TRUE; 0 IS FALSE

percfront = set((1, c) for c in range(gridx))

# normalize row to [0,1]
row_scale = np.linspace(0, 1, gridy)


max_steps = gridx * gridy  # upper cap for steps taken
step = 0

while step < (gridx * gridy) and len(percfront) > 0:
    # compute efective values for frontier sites and choose min
    percfrontlist = list(percfront)

    rows = np.array([p[0] for p in percfrontlist], dtype=int)
    cols = np.array([p[1] for p in percfrontlist], dtype=int)
    #t_x = thresholds[rows, cols] - gravityvalue*row_scale[rows]

    t_x = []
    for (temprow, tempcolumn) in percfrontlist:
        tvalue = thresholds[temprow, tempcolumn] - gravityvalue * row_scale[temprow]
        t_x.append(tvalue)
    t_x = np.array(t_x)

    # pick candidate with minimal effective theshold

    currentrow, currentcolumn = percfrontlist[int(np.argmin(t_x))]

    # invade it
    invaded[currentrow, currentcolumn] = 1
    percfront.remove((currentrow, currentcolumn))

    if currentrow == gridy - 1:
        print(f"Percolated to bottom at step {step}.")
        break

    # add its uninvaded neighbors to fronteir
    for (temprow, tempcolumn) in neighbouringsites(currentrow, currentcolumn, gridy-1, gridx-1):
        if invaded[temprow, tempcolumn]:
            pass
        else:
            percfront.add((temprow, tempcolumn))

    step += 1


    # Stop early if invasion reached bottom row (percolation algrth complete)
    for i in range(gridx):
        if invaded[-1, i] == 1:
            print(f"Percolated to bottom at step {step}.")
            break


def plot_cluster(invaded, background):  

    gridy, gridx = invaded.shape
    plt.figure(figsize=(7,7/5*7))   #figsize=(6.1,6.1/5*8)

    # show background only where not invaded (so invaded stands out)
    display = np.where(invaded, np.nan, background)

    im = plt.imshow(display, origin='upper', interpolation='none', aspect='auto')


    cbar = plt.colorbar(im, label='threshold (uninvaded sites)')
    cbar.ax.tick_params(labelsize=35)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar.set_label('Effective threshold (uninvaded sites)', fontsize=30)


    # overlay invaded sites
    ys, xs = np.where(invaded)
    plt.scatter(xs, ys, s=1, c='k')


    heights = np.full(gridx, np.nan)

    for col in range(gridx):
        invaded_rows = np.where(invaded[:, col])[0]
        if invaded_rows.size > 0:
            heights[col] = invaded_rows.max()
    cols = np.arange(gridx)
    
    plt.plot(cols, heights, linewidth=3.0, label='front', color='red')
    #plt.legend(loc='upper centre', fontsize=25)


    plt.xlabel('x (column)', fontsize=35)
    #plt.ylabel('y (row, 0 = top)', fontsize=35)
    plt.title('Invasion percolation \n cluster and front (g=2)', fontsize=35)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tight_layout()
    plt.show()


effective_field = thresholds - gravityvalue * row_scale[:, None]
plot_cluster(invaded, background=effective_field)
