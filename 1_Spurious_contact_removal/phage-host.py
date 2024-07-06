import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

class Phage_Host:
    def __init__(self, contig_info_path, contact_matrix_path):
        self.contig_info_path = contig_info_path
        self.contact_matrix_path = contact_matrix_path
        self.contig_info, self.contact_matrix = self.load_data()
    
    def load_data(self):
        contig_info = pd.read_csv(self.contig_info_path)
        npzfile = np.load(self.contact_matrix_path)
        sparse_matrix = csr_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])
        contact_matrix = sparse_matrix.toarray()
        return contig_info, contact_matrix

    def calculate_connectivity_density(self, id1, id2):
        return self.contact_matrix[id1, id2] / (len(self.contact_matrix) ** 2)

    def first_round_filter(self, min_links, connectivity_ratio, intra_mag_links):
        filtered_pairs = []
        contig_names = self.contig_info['Contig name'].tolist()
        contig_coverage = self.contig_info['Contig coverage'].tolist()
        
        for phage_id in range(self.contact_matrix.shape[0]):
            for host_id in range(phage_id + 1, self.contact_matrix.shape[1]):
                if self.contact_matrix[phage_id, host_id] == 0:
                    continue
                
                phage_name = contig_names[phage_id]
                host_name = contig_names[host_id]
                
                hi_c_links = self.contact_matrix[phage_id, host_id]
                intra_mag = self.contact_matrix[host_id, host_id]
                
                if hi_c_links < min_links or intra_mag < intra_mag_links:
                    self.contact_matrix[phage_id, host_id] = 0
                    continue
                
                D_VH = self.calculate_connectivity_density(phage_id, host_id)
                D_H = self.calculate_connectivity_density(host_id, host_id)
                
                H = contig_coverage[host_id]
                V = contig_coverage[phage_id]
                L = hi_c_links
                L_v = np.sum(self.contact_matrix[phage_id])
                
                R_prime = (D_VH / D_H) * (H * L_v / (V * L))
                
                if R_prime >= connectivity_ratio:
                    filtered_pairs.append((phage_name, host_name))
                else:
                    self.contact_matrix[phage_id, host_id] = 0
        
        return filtered_pairs

    def categorize_data_for_roc_analysis(self, min_value, max_value, increment):
        print("Generating ROC categories")
        roc_data = dict()
        threshold_values = list()
        threshold = min_value
        while threshold < max_value:
            roc_data[threshold] = [0, set()]
            threshold_values.append(threshold)
            threshold *= increment
        contig_names = self.contig_info['Contig name'].tolist()
        for phage_id in range(self.contact_matrix.shape[0]):
            for host_id in range(phage_id + 1, self.contact_matrix.shape[1]):
                if self.contact_matrix[phage_id, host_id] == 0:
                    continue
                copy_count = self.contact_matrix[phage_id, host_id]
                for threshold in threshold_values:
                    if threshold > copy_count:
                        break
                    roc_data[threshold][0] += 1
                    roc_data[threshold][1].add(contig_names[phage_id])
        for threshold in threshold_values:
            roc_data[threshold][1] = len(roc_data[threshold][1])
        return roc_data, threshold_values

    def generate_roc_curve_values(self, roc_data, threshold_values, min_value):
        true_positives = list()
        false_positives = list()
        max_possible_hits_accepted = roc_data[min_value][0]
        max_possible_mobile_contigs_with_hosts = roc_data[min_value][1]
        for threshold in threshold_values:
            hits_accepted = roc_data[threshold][0]
            false_positives.append(hits_accepted / max_possible_hits_accepted)
            mobile_contigs_with_hosts = roc_data[threshold][1]
            true_positives.append(mobile_contigs_with_hosts / max_possible_mobile_contigs_with_hosts)
        return true_positives, false_positives

    def get_optimal_threshold(self, threshold_values, true_positives, false_positives, min_fraction_without_hosts=0.8):
        optimal_threshold = 0
        fp_rate = 0
        tp_rate = 0
        for i, threshold in enumerate(threshold_values):
            optimal_threshold = threshold
            fp_rate = false_positives[i]
            tp_rate = true_positives[i]
            if fp_rate + tp_rate < 1 or tp_rate < min_fraction_without_hosts:
                break
        return optimal_threshold, fp_rate, tp_rate

    def second_round_filter(self, filtered_pairs, optimal_threshold, avg_count_ratio):
        refined_pairs = []
        contig_names = self.contig_info['Contig name'].tolist()
        
        for phage_name, host_name in filtered_pairs:
            phage_id = contig_names.index(phage_name)
            host_id = contig_names.index(host_name)
            
            avg_counts = np.mean(self.contact_matrix[phage_id, host_id])
            max_counts = np.max(self.contact_matrix[phage_id])
            
            if avg_counts >= avg_count_ratio * max_counts and avg_counts >= optimal_threshold:
                refined_pairs.append((phage_name, host_name))
            else:
                self.contact_matrix[phage_id, host_id] = 0.01
        return refined_pairs

    def save_filtered_results(self, output_npz):
        sparse_filtered_matrix = csr_matrix(self.contact_matrix)
        np.savez(output_npz, data=sparse_filtered_matrix.data, indices=sparse_filtered_matrix.indices, indptr=sparse_filtered_matrix.indptr, shape=sparse_filtered_matrix.shape)

    def main(self, min_links=2, connectivity_ratio=0.1, intra_mag_links=10, avg_count_ratio=0.8, min_value=0.0001, max_value=100, increment=1.5, output_npz='../0_Documents/PH_contact_matrix.npz'):
        filtered_pairs = self.first_round_filter(min_links, connectivity_ratio, intra_mag_links)

        roc_data, threshold_values = self.categorize_data_for_roc_analysis(min_value, max_value, increment)
        true_positives, false_positives = self.generate_roc_curve_values(roc_data, threshold_values, min_value)
        optimal_threshold, fp_rate, tp_rate = self.get_optimal_threshold(threshold_values, true_positives, false_positives)

        print("Optimal Threshold:", optimal_threshold)

        refined_pairs = self.second_round_filter(filtered_pairs, optimal_threshold, avg_count_ratio)

        print(refined_pairs)

        self.save_filtered_results(output_npz)

if __name__ == "__main__":
    contig_info_path = '../0_Documents/contig_information.csv'
    contact_matrix_path = '../0_Documents/raw_contact_matrix.npz'

    phage_host = Phage_Host(contig_info_path, contact_matrix_path)
    phage_host.main()