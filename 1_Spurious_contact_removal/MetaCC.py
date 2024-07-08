import numpy as np
from math import log, exp, sqrt
import pandas as pd
import statsmodels.api as sm
import scipy.sparse as sp


class MetaCC:
    def __init__(self, path, contig_info, seq_map, norm_result, threshold):
        self.path = path
        self.seq_map_raw = seq_map
        self.seq_map = seq_map
        self.norm_result = norm_result
        self.threshold = threshold
        self.name = []
        self.sites = []
        self.length = []
        self.coverage = []

        for temp in contig_info:
            self.name.append(temp['name'])
            self.sites.append(temp['sites'])
            self.length.append(temp['length'])
            self.coverage.append(temp['coverage'])

        del contig_info
        
        self.name = np.array(self.name)
        self.sites = np.array(self.sites)
        self.length = np.array(self.length)
        self.coverage = np.array(self.coverage)
        

        self.norm()

    def load(contig_file):
        names = ['name', 'sites', 'length', 'coverage', 'signal']
        df = pd.read_csv(contig_file, usecols=range(5), header=None, skiprows=1, names=names)
        
        df['sites'] = pd.to_numeric(df['sites'])
        df['length'] = pd.to_numeric(df['length'])
        df['coverage'] = pd.to_numeric(df['coverage'])
        df['signal'] = pd.to_numeric(df['signal'])
        
        df['sites'] = np.log(df['sites'])
        df['length'] = np.log(df['length'])
        df['coverage'] = np.log(df['coverage'])
        
        exog = df[['sites', 'length', 'coverage']]
        endog = df[['signal']]
        exog = sm.add_constant(exog)
        
        glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=1))
        res = glm_nb.fit(method="lbfgs")
        norm_result = res.params.tolist()
        
        contig_info = [
            {'name': row['name'], 'sites': np.exp(row['sites']), 'length': np.exp(row['length']), 'coverage': np.exp(row['coverage'])}
            for _, row in df.iterrows()
        ]
        
        return contig_info, norm_result      

    def norm(self):
        self.seq_map = self.seq_map.tocoo()
        _map_row = self.seq_map.row
        _map_col = self.seq_map.col
        _map_data = self.seq_map.data
        _index = _map_row < _map_col
        _map_row = _map_row[_index]
        _map_col = _map_col[_index]
        _map_data = _map_data[_index]
        
        _map_coor = list(zip(_map_row, _map_col, _map_data))
        coeff = self.norm_result
        
        self.seq_map = self.seq_map.tolil()
        self.seq_map = self.seq_map.astype(np.float64)
        
        mu_vector = []
        for contig_feature in zip(self.sites, self.length, self.coverage):
            mu_vector.append(exp(coeff[0] + coeff[1]*log(contig_feature[0])+ coeff[2]*log(contig_feature[1])+ coeff[3]*log(contig_feature[2])))
        scal = np.max(mu_vector)
        _norm_contact = []
        
        for i in _map_coor:
            x = i[0]
            y = i[1]
            d = i[2]
            
            d_norm = scal * d / sqrt(mu_vector[x] * mu_vector[y])
            _norm_contact.append(d_norm)
            
            self.seq_map[x, y] = d_norm
            self.seq_map[y, x] = d_norm
            
        
        # Remove spurious contacts
        cutoffs = np.percentile(_norm_contact, self.threshold * 100)
        count = 0
        for j in range(len(_norm_contact)):
            x = _map_row[j]
            y = _map_col[j]
            if _norm_contact[j] < cutoffs:
                self.seq_map[x, y] = 0
                self.seq_map[y, x] = 0
                count += 1
        
        del _map_row, _map_col, _map_data, _map_coor, _norm_contact, count

    def save(self, output_path):
        self.seq_map = self.seq_map.tocsr()
        sp.save_npz(output_path, self.seq_map)

    def main(contig_info_path, raw_contact_matrix_path, threshold, output_path):
        contig_info, norm_result = MetaCC.load(contig_info_path)
        raw_contact_matrix = sp.load_npz(raw_contact_matrix_path)
        norm_cc_map = MetaCC(path=raw_contact_matrix_path, contig_info=contig_info, seq_map=raw_contact_matrix, norm_result=norm_result, threshold=threshold)
        norm_cc_map.save(output_path)


if __name__ == "__main__":
    contig_info_path = '../0_Documents/contig_information.csv'
    raw_contact_matrix_path = '../0_Documents/raw_contact_matrix.npz'
    threshold = 0.05 
    output_path = '../0_Documents/MetaCC_contact_matrix.npz'
    MetaCC.main(contig_info_path, raw_contact_matrix_path, threshold, output_path)