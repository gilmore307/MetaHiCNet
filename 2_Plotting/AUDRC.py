import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def load_data(file_path):
    npzfile = np.load(file_path)
    data = npzfile['data']
    indices = npzfile['indices']
    indptr = npzfile['indptr']
    shape = npzfile['shape']
    contact_matrix = csr_matrix((data, indices, indptr), shape=shape)
    return contact_matrix

def plot_audrc(contact_matrix, label):
    intra_species_matrix = csr_matrix((contact_matrix.diagonal(), (np.arange(contact_matrix.shape[0]), np.arange(contact_matrix.shape[0]))), shape=contact_matrix.shape)

    spurious_matrix = contact_matrix.copy()
    spurious_matrix.setdiag(0)

    total_intra_species = intra_species_matrix.count_nonzero()
    total_spurious = spurious_matrix.count_nonzero()

    percentiles = np.percentile(contact_matrix, np.linspace(0, 100, 1000))

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

    plt.plot(proportion_discarded_spurious, proportion_retained_intra_species, label=f'{label} (AUDRC: {audrc:.3f})')
    #plt.scatter(proportion_discarded_spurious, proportion_retained_intra_species, s=10, label=f'{label} Points')

    return audrc

def extract_label(file_path):
    base_name = os.path.basename(file_path)
    label = base_name.split('_contact_matrix.npz')[0]
    return label

plt.figure(figsize=(10, 6))

datasets = ['../0_Documents/Raw_contact_matrix.npz', '../0_Documents/Cutoff_contact_matrix.npz', '../0_Documents/PH_contact_matrix.npz',
            '../0_Documents/HiCzin_contact_matrix.npz', '../0_Documents/MetaCC_contact_matrix.npz']

for dataset in datasets:
    contact_matrix = load_data(dataset)
    label = extract_label(dataset)
    plot_audrc(contact_matrix, label)

plt.xlabel('Proportion of discarded spurious contacts')
plt.ylabel('Proportion of retained intra-species contacts')
plt.title('Discard-Retain Curve for Multiple Datasets (Percentile-based Thresholds)')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
plt.savefig('discard_retain_curve.png')