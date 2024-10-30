import numpy as np
from scipy.sparse import csc_matrix, coo_matrix

bin_contact_matrix_path= 'output/bin_contact_matrix.npz'
bin_contact_matrix_data = np.load(bin_contact_matrix_path)
data = bin_contact_matrix_data['data']
indices = bin_contact_matrix_data['indices']
indptr = bin_contact_matrix_data['indptr']
shape = bin_contact_matrix_data['shape']
sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)

coo_matrix_format = sparse_matrix.tocoo()

data = coo_matrix_format.data
row = coo_matrix_format.row
col = coo_matrix_format.col
shape = coo_matrix_format.shape

np.savez_compressed('output/bin_contact_matrix.npz', data=data, row=row, col=col, shape=shape)