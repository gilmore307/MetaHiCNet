import numpy as np
from scipy.sparse import coo_matrix

bin_contact_matrix_path= 'output/bin_contact_matrix.npz'
bin_contact_matrix_data = np.load(bin_contact_matrix_path)
data = bin_contact_matrix_data['data']
row = bin_contact_matrix_data['row']
col = bin_contact_matrix_data['col']
shape = tuple(bin_contact_matrix_data['shape'])

contact_matrix = coo_matrix((data, (row, col)), shape=shape)
contact_matrix = contact_matrix.toarray()
