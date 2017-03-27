import os
import numpy as np
import math

root = '/Users/apple/Desktop/CHL/CS Study/CZ4041/OpportunityUCIDataset/dataset/'
#out_root = '/Users/apple/Desktop/CHL/CS Study/CZ4041/OpportunityUCIDataset/output/'
filename_pre='S1-ADL1.dat'

filename = 'S1-Drill.dat'
data_root = os.path.join(root,filename_pre)
f = open(data_root, 'r')
raw_array = np.fromfile(f, dtype=float, sep=' ')

mask_window_L = 134
mask_window_H = 242

#print(raw_array.size)

m = np.reshape(raw_array, (-1,250))

mask = [i for i in range(mask_window_L,mask_window_H+1)]
mask.extend([i for i in range(244,249)])
m = np.delete(m,mask,1)
print(m.shape)

#out_filename = 'dataset' + "-drill" + '.npy'
#out_data_root = os.path.join(out_root, out_filename)

#np.save(out_data_root, m)
print("Saving complete seq.")








    
