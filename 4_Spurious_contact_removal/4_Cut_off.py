import numpy as np
from scipy.sparse import csr_matrix

class Cutoff:
    def __init__(self, file_path, threshold):
        self.file_path = file_path
        self.threshold = threshold
        self.matrix = self.load_data()
        self.dense_matrix = self.matrix.toarray()
    
    def load_data(self):
        data = np.load(self.file_path)
        indices = data['indices']
        indptr = data['indptr']
        shape = tuple(data['shape'])
        data_values = data['data']
        sparse_matrix = csr_matrix((data_values, indices, indptr), shape=shape)
        return sparse_matrix
    
    def apply_threshold(self):
        self.dense_matrix[self.dense_matrix < self.threshold] = 0
        self.modified_matrix = csr_matrix(self.dense_matrix)
    
    def save_matrix(self, output_path):
        np.savez(output_path, 
                 data=self.modified_matrix.data, 
                 indices=self.modified_matrix.indices, 
                 indptr=self.modified_matrix.indptr, 
                 shape=self.modified_matrix.shape)

    def main():
        file_path = '../0_Documents/raw_contact_matrix.npz'
        threshold = 2
        output_path = '../0_Documents/Cutoff_contact_matrix.npz'
        
        cutoff = Cutoff(file_path, threshold)
        cutoff.apply_threshold()
        cutoff.save_matrix(output_path)

if __name__ == "__main__":
    Cutoff.main()