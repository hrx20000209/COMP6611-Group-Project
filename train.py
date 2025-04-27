import numpy as np

file = np.load('/home/hrx/Data/6611_Data/squating/sample_00002.npz', allow_pickle=True)
print(file['radar'])