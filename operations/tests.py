import numpy as np

radius = 3
ts_len = 100
indices = [1, 34, 77, 89, 98]

indices_to_remove = []

# Build the list of indices to remove using the selected indices and radius
for index in indices:
    index_range = list(np.arange(index-radius, index+radius+1, 1))
    for i in index_range:
        indices_to_remove.append(i)

indices_to_remove = np.asarray(indices_to_remove)
indices_to_remove = np.delete(indices_to_remove, np.where(~(0<=indices_to_remove)))
indices_to_remove3 = np.delete(indices_to_remove, np.where(~(indices_to_remove<ts_len)))


print(indices_to_remove)
print(indices_to_remove3)