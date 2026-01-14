import numpy as np
import matplotlib.pyplot as plt

L = 141              # system grid size
M = 500              # number of realisations
a = 0.5        # reference value (≈ pc)
g = 1           # total gradient across system (tune this)

for i in range(M):
    
    # make random field
    grid = np.random.rand(L, L)
    
    # vertical coordinate (0 at top, 1 at bottom)
    y = np.linspace(0, 1, L)
    
    # probability function
    p_y = a + g * (y - 0.5)
    
    # build occupation grid
    occ = np.zeros((L, L), dtype=bool)
    for j in range(L):
        occ[j, :] = grid[j, :] < p_y[j] # vectorised code to find which squares should be coloured

plt.figure(figsize=(11*0.9, 10*0.9))
plt.imshow(occ, origin='lower', cmap='binary')
plt.xlabel('x', fontsize=40)
plt.ylabel('y', fontsize=40)
plt.title('Gradient site percolation', fontsize=40)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
plt.tight_layout()
plt.show()
