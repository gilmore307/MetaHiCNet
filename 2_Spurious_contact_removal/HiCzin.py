import numpy as np
import pandas as pd
import scipy.sparse as scisp
from math import log, exp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HiCzin:
    def __init__(self, contact_matrix_path, contig_info_path, output_path, thres):
        self.contact_matrix_path = contact_matrix_path
        self.contig_info_path = contig_info_path
        self.output_path = output_path
        self.thres = thres
        self.contig_info = None
        self.contact_matrix = None

    def load_data(self):
        contig_info = pd.read_csv(self.contig_info_path)
        contig_info = contig_info.rename(columns={
            'Contig name': 'contig_name',
            'Number of restriction sites': 'site',
            'Contig length': 'length',
            'Contig coverage': 'coverage'
        })
        self.contig_info = contig_info[['contig_name', 'site', 'length', 'coverage']]

        npzfile = np.load(self.contact_matrix_path)
        sparse_matrix = scisp.csr_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])
        self.contact_matrix = sparse_matrix.tocoo()

    def generate_valid_contact_file(self):
        coo = self.contact_matrix
        valid_contact = np.vstack((coo.row, coo.col, coo.data)).T
        valid_contact_df = pd.DataFrame(valid_contact, columns=['index1', 'index2', 'contacts'])

        valid_contact_df['index1'] = valid_contact_df['index1'].astype(int)
        valid_contact_df['index2'] = valid_contact_df['index2'].astype(int)
        return valid_contact_df

    def normalize_and_fit_model(self, valid_contact_df, thres):
        sample_data = valid_contact_df
        contig_info = self.contig_info
        
        contig_info['site'] = contig_info['site'].replace(0, 1)
        contig_info.loc[contig_info['coverage'] == 0, 'coverage'] = contig_info[contig_info['coverage'] != 0]['coverage'].min()
        
        sample_site = np.log(contig_info.loc[sample_data['index1'], 'site'].values * contig_info.loc[sample_data['index2'], 'site'].values)
        sample_len = np.log(contig_info.loc[sample_data['index1'], 'length'].values * contig_info.loc[sample_data['index2'], 'length'].values)
        sample_cov = np.log(contig_info.loc[sample_data['index1'], 'coverage'].values * contig_info.loc[sample_data['index2'], 'coverage'].values)
        sampleCon = sample_data['contacts'].to_numpy()

        mean_site = np.mean(sample_site)
        sd_site = np.std(sample_site)
        mean_len = np.mean(sample_len)
        sd_len = np.std(sample_len)
        mean_cov = np.mean(sample_cov)
        sd_cov = np.std(sample_cov)

        sample_site = (sample_site - mean_site) / sd_site
        sample_len = (sample_len - mean_len) / sd_len
        sample_cov = (sample_cov - mean_cov) / sd_cov

        data_sample = pd.DataFrame({'sample_site': sample_site, 'sample_len': sample_len, 'sample_cov': sample_cov, 'sampleCon': sampleCon})

        model = smf.glm(formula='sampleCon ~ sample_site + sample_len + sample_cov', 
                data=data_sample, 
                family=sm.families.NegativeBinomial(alpha=1))
        
        fit = model.fit(method="lbfgs")
        
        print(fit.summary())

        coeff = fit.params.to_numpy()

        res_sample = sampleCon / np.exp(coeff[0] + coeff[1]*sample_site + coeff[2]*sample_len + coeff[3]*sample_cov)

        index_nonzero = res_sample > 0
        res_sample_nonzero = res_sample[index_nonzero]
        
        perc = np.percentile(res_sample_nonzero, thres * 100)

        result = np.concatenate([coeff[:4], [perc, mean_site, sd_site, mean_len, sd_len, mean_cov, sd_cov]])
        print("coeff: ", coeff[:4], "perc: ", perc, "site: ", mean_site, sd_site, "len: ", mean_len, sd_len, "cov: ", mean_cov, sd_cov)
        return result 
    
    def norm_contact_matrix(self, norm_result):
        seq_map = self.contact_matrix
        seq_map = seq_map.tocoo()
        contig_info = self.contig_info
        
        _map_row = seq_map.row
        _map_col = seq_map.col
        _map_data = seq_map.data
        _map_coor = list(zip(_map_row, _map_col, _map_data))
        coeff = norm_result[0:4]

        seq_map = seq_map.tolil()
        seq_map = seq_map.astype(np.float64)
        for x, y, d in _map_coor:
            s1 = contig_info.loc[x, 'site']
            if s1 == 0:
                s1 = 1
            s2 = contig_info.loc[y, 'site']
            if s2 == 0:
                s2 = 1
            s = (log(s1 * s2) - norm_result[5]) / norm_result[6]

            l1 = contig_info.loc[x, 'length']
            l2 = contig_info.loc[y, 'length']
            l = (log(l1 * l2) - norm_result[7]) / norm_result[8]

            c1 = contig_info.loc[x, 'coverage']
            c2 = contig_info.loc[y, 'coverage']
            c = (log(c1 * c2) - norm_result[9]) / norm_result[10]

            d_norm = d / exp(coeff[0] + coeff[1] * s + coeff[2] * l + coeff[3] * c)
            
            if d_norm > norm_result[4]:
                seq_map[x, y] = d_norm
            else:
                seq_map[x, y] = 0
        
        seq_map_csr = seq_map.tocsr()
        normalized_matrix_path = self.output_path
        scisp.save_npz(normalized_matrix_path, seq_map_csr)
        logger.info(f"Normalized contact matrix saved to {normalized_matrix_path}")
        
    def main(self):
        self.load_data()
        valid_contact_df = self.generate_valid_contact_file()
        norm_result = self.normalize_and_fit_model(valid_contact_df, self.thres)
        self.norm_contact_matrix(norm_result)

if __name__ == '__main__':
    contact_matrix_path = '../0_Documents/raw_contact_matrix.npz'
    contig_info_path = '../0_Documents/contig_information.csv'
    output_path = '../0_Documents/HiCzin_contact_matrix.npz'
    thres=0.05

    hiczin = HiCzin(contact_matrix_path, contig_info_path, output_path, thres)
    hiczin.main()
