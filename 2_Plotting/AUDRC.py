import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import os

def load_data(contact_matrix_file, csv_file_path):
    npzfile = np.load(contact_matrix_file)
    data = npzfile['data']
    indices = npzfile['indices']
    indptr = npzfile['indptr']
    shape = npzfile['shape']
    contact_matrix = csr_matrix((data, indices, indptr), shape=shape)

    contig_info_df = pd.read_csv(csv_file_path)
    annotations = contig_info_df['Contig annotation'].values

    return contact_matrix, annotations

def prep_matrix(contact_matrix, annotations):
    annotation_to_index = {annotation: np.where(annotations == annotation)[0] for annotation in np.unique(annotations)}

    intra_species_mask = np.zeros(contact_matrix.shape, dtype=bool)
    for indices in annotation_to_index.values():
        intra_species_mask[np.ix_(indices, indices)] = True

    intra_species_matrix = csr_matrix(contact_matrix.multiply(intra_species_mask))

    spurious_matrix = csr_matrix(contact_matrix.multiply(~intra_species_mask))

    return intra_species_matrix, spurious_matrix

def calculate_audrc(contact_matrix, intra_species_matrix, spurious_matrix):
    total_intra_species = intra_species_matrix.count_nonzero()
    total_spurious = spurious_matrix.count_nonzero()

    percentiles = np.percentile(contact_matrix.data, np.linspace(0, 100, 1000))

    proportion_discarded_spurious = []
    proportion_retained_intra_species = []

    for threshold in percentiles:
        retained_intra_species = (intra_species_matrix.data >= threshold).sum()

        discarded_spurious = (spurious_matrix.data > 0) & (spurious_matrix.data < threshold)
        discarded_spurious_count = discarded_spurious.sum()

        prop_retained_intra_species = retained_intra_species / total_intra_species
        prop_discarded_spurious = discarded_spurious_count / total_spurious

        proportion_discarded_spurious.append(prop_discarded_spurious)
        proportion_retained_intra_species.append(prop_retained_intra_species)

    proportion_discarded_spurious = np.array(proportion_discarded_spurious)
    proportion_retained_intra_species = np.array(proportion_retained_intra_species)

    audrc = np.trapz(proportion_retained_intra_species, proportion_discarded_spurious)
    
    return proportion_discarded_spurious, proportion_retained_intra_species, audrc

def extract_label(file_path):
    base_name = os.path.basename(file_path)
    label = base_name.split('_contact_matrix.npz')[0]
    return label

datasets = [
    '../0_Documents/Raw_contact_matrix.npz', 
    '../0_Documents/Cutoff_contact_matrix.npz', 
    '../0_Documents/PH_contact_matrix.npz',
    '../0_Documents/HiCzin_contact_matrix.npz', 
    '../0_Documents/MetaCC_contact_matrix.npz'
]
csv_file_path = '../0_Documents/contig_information.csv'

plt.figure(figsize=(12, 8))

for dataset in datasets:
    label = extract_label(dataset)
    contact_matrix, annotations = load_data(dataset, csv_file_path)
    intra_species_matrix, spurious_matrix = prep_matrix(contact_matrix, annotations)
    proportion_discarded_spurious, proportion_retained_intra_species, audrc = calculate_audrc(contact_matrix, intra_species_matrix, spurious_matrix)
    
    plt.plot(proportion_discarded_spurious, proportion_retained_intra_species, label=f'{label} (AUDRC: {audrc:.3f})')
    #plt.scatter(proportion_discarded_spurious, proportion_retained_intra_species, s=10, label=f'{label} Data Points')

plt.xlabel('Proportion of discarded spurious contacts')
plt.ylabel('Proportion of retained intra-species contacts')
plt.title('Discard-Retain Curve for All Datasets (Percentile-based Thresholds)')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
