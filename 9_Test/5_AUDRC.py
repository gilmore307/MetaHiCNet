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
    intra_species_array = intra_species_matrix.data[intra_species_matrix.data != 0]

    spurious_matrix = csr_matrix(contact_matrix.multiply(~intra_species_mask))
    spurious_array = spurious_matrix.data[spurious_matrix.data != 0]

    return intra_species_array, spurious_array

def calculate_audrc(contact_matrix, intra_species_array, spurious_array):
    total_intra_species = len(intra_species_array)
    total_spurious = len(spurious_array)

    percentiles = np.percentile(contact_matrix.data, np.linspace(0, 100, 1000))

    proportion_discarded_spurious = []
    proportion_retained_intra_species = []

    for threshold in percentiles:
        retained_intra_species = (intra_species_array >= threshold).sum()
        discarded_spurious = (spurious_array < threshold).sum()

        prop_retained_intra_species = retained_intra_species / total_intra_species
        prop_discarded_spurious = discarded_spurious / total_spurious

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

results = []

for dataset in datasets:
    label = extract_label(dataset)
    contact_matrix, annotations = load_data(dataset, csv_file_path)
    intra_species_array, spurious_array = prep_matrix(contact_matrix, annotations)
    proportion_discarded_spurious, proportion_retained_intra_species, audrc = calculate_audrc(contact_matrix, intra_species_array, spurious_array)
    
    results.append((label, proportion_discarded_spurious, proportion_retained_intra_species, audrc))

results.sort(key=lambda x: x[3], reverse=True)

plt.figure(figsize=(12, 8))

for label, proportion_discarded_spurious, proportion_retained_intra_species, audrc in results:
    plt.plot(proportion_discarded_spurious, proportion_retained_intra_species, label=f'{label} (AUDRC: {audrc:.3f})')

plt.xlabel('Proportion of discarded spurious contacts')
plt.ylabel('Proportion of retained intra-species contacts')
plt.title('Discard-Retain Curve for All Datasets (Percentile-based Thresholds)')
plt.legend()
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('AUDRC.png')
plt.show()


def heatmap(contact_matrix):
    """
    Generates a heatmap from a contact matrix with downsampling to 100 bins and logarithmic scaling.

    Parameters:
    - contact_matrix (array or sparse matrix): The contact matrix in dense or sparse format.
    """
    # If the contact matrix is in sparse format, convert it to a dense array
    if isinstance(contact_matrix, coo_matrix):
        contact_matrix = contact_matrix.toarray()

    # Downsample the matrix to 100 bins
    def downsample_matrix(matrix, target_bins=100):
        original_size = matrix.shape[0]
        factor = original_size // target_bins  # Calculate downsampling factor
        return matrix[:factor * target_bins, :factor * target_bins].reshape(target_bins, factor, target_bins, factor).mean(axis=(1, 3))

    downsampled_matrix = downsample_matrix(contact_matrix)

    # Apply logarithmic scaling
    log_scaled_matrix = np.log1p(downsampled_matrix)  # log(1 + x) to handle zeros

    # Plot the heatmap with a wider color spectrum
    plt.figure(figsize=(10, 8))
    sns.heatmap(log_scaled_matrix, cmap='plasma', cbar=False, xticklabels=False, yticklabels=False, square=True)
    plt.show()

# Example call to the function (replace 'contact_matrix' with an actual matrix when running in a live environment)
# heatmap(contact_matrix)

