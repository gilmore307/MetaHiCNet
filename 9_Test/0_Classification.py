import numpy as np
from scipy.sparse import csc_matrix

bin_contact_matrix_path= 'input/Raw_contact_matrix.npz'
bin_contact_matrix_data = np.load(bin_contact_matrix_path)
data = bin_contact_matrix_data['data']
indices = bin_contact_matrix_data['indices']
indptr = bin_contact_matrix_data['indptr']
shape = bin_contact_matrix_data['shape']
sparse_matrix = csc_matrix((data, indices, indptr), shape=shape)
dense_matrix = sparse_matrix.toarray()