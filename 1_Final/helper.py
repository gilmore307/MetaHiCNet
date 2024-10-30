import os
import io
import numpy as np
import base64
from scipy.sparse import csr_matrix
import requests
import py7zr
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.sparse import coo_matrix, spdiags
import statsmodels.api as sm

logger = logging.getLogger("app_logger")

def save_file_to_user_folder(contents, filename, user_folder, folder_name='output'):

    # Ensure the user folder exists
    user_folder_path = os.path.join('assets', folder_name, user_folder)
    os.makedirs(user_folder_path, exist_ok=True)
    
    # Define the full file path
    file_path = os.path.join(user_folder_path, filename)

    # Decode contents
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        # Save as CSV file
        if filename.endswith('.csv'):
            with open(file_path, 'wb') as file:
                file.write(decoded)
            print(f"CSV file saved: {file_path}")

        # Save as NPZ file in COO format
        elif filename.endswith('.npz'):
            npz_file = np.load(io.BytesIO(decoded))
            if all(key in npz_file for key in ['data', 'row', 'col', 'shape']):
                # Already in COO format
                coo = coo_matrix((npz_file['data'], (npz_file['row'], npz_file['col'])), shape=tuple(npz_file['shape']))
                np.savez_compressed(file_path, data=coo.data, row=coo.row, col=coo.col, shape=coo.shape)
                print(f"COO NPZ file saved: {file_path}")
            else:
                # Save the raw npz content if not sparse
                with open(file_path, 'wb') as file:
                    file.write(decoded)
                print(f"Raw NPZ file saved: {file_path}")

        # Save as 7z archive
        elif filename.endswith('.7z'):
            with open(file_path, 'wb') as file:
                file.write(decoded)
            print(f"7z file saved: {file_path}")

        else:
            raise ValueError("Unsupported file format. Only .csv, .npz, and .7z are supported.")

    except Exception as e:
        print(f"Failed to save file {filename}: {str(e)}")

    return file_path

# a_preparation
def query_plasmid_id(ids, fasta=False):
    # variables
    URL = 'https://ccb-microbe.cs.uni-saarland.de/plsdb/plasmids/api/'

    # logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%y:%m:%d %H:%M:%S'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # CPU cores
    core = os.cpu_count()
    max_workers = core * 4

    def fetch_plasmid_data(plasmid_id, session):
        """
        Function to fetch data for a single plasmid ID using a session object.
        """
        PARAMS = {'plasmid_id': plasmid_id}
        try:
            response = session.get(url=URL + 'plasmid/', params=PARAMS)
            response.raise_for_status()
            d = response.json()
            if 'found' in d:
                d['found']['searched'] = plasmid_id
                d['found']['label'] = 'found'
                return d['found'], None
            elif 'searched' in d:
                d['label'] = 'notfound'
                return None, d
            else:
                return None, None
        except requests.exceptions.RequestException:
            return None, None

    if not isinstance(ids, list):
        ids = ids.split()
    
    assert len(ids) > 0, "List of ids is empty."

    found = []
    notfound = []
    fastas = []

    # Use a session object for efficiency
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_id = {executor.submit(fetch_plasmid_data, plasmid_id, session): plasmid_id for plasmid_id in ids}
            
            for future in tqdm(as_completed(future_to_id), total=len(future_to_id)):
                try:
                    result, not_found_result = future.result()
                    if result:
                        found.append(result)
                        if fasta:
                            fastas.append(result['searched'])
                    elif not_found_result:
                        notfound.append(not_found_result)
                except Exception:
                    pass  # Silence exceptions here

    logger.info('Search is finished')
    logger.info(f'{len(found)} of {len(ids)} ids were found')

    # convert arrays to pd dataframe and merge results
    df1 = pd.DataFrame(data=found)
    df2 = pd.DataFrame(data=notfound)
    df = pd.concat([df1, df2], ignore_index=True, sort=False)

    return df

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename:
        # Load CSV file
        return pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    elif 'npz' in filename:
        # Load the npz file
        npzfile = np.load(io.BytesIO(decoded))
        
        # Check for COO matrix keys
        if all(key in npzfile for key in ['data', 'row', 'col', 'shape']):
            data = npzfile['data']
            row = npzfile['row']
            col = npzfile['col']
            shape = tuple(npzfile['shape'])
            
            # Reconstruct the sparse matrix in COO format
            contact_matrix = coo_matrix((data, (row, col)), shape=shape)
            return contact_matrix
        
        else:
            raise ValueError("The matrix file does not contain the expected COO matrix keys.")
    
    elif '7z' in filename:
        # Load and extract .7z archive
        with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as z:
            return z.getnames()
    
    else:
        raise ValueError("Unsupported file format.")


def get_file_size(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    size_in_bytes = len(decoded)
    size_in_kb = size_in_bytes / 1024
    return f"{size_in_kb:.2f} KB"

def validate_csv(df, required_columns, optional_columns=[]):
    all_columns = required_columns + optional_columns
    if not all(column in df.columns for column in all_columns):
        missing_cols = set(required_columns) - set(df.columns)
        logger.error(f"Missing columns in the CSV file: {missing_cols}")
        raise ValueError(f"Missing columns in the file: {missing_cols}")
    for col in required_columns:
        if df[col].isnull().any():
            logger.error(f"Required column '{col}' has missing values.")
            raise ValueError(f"Required column '{col}' has missing values.")
    return True

def validate_contig_matrix(contig_data, contact_matrix):
    num_contigs = len(contig_data)

    # Check if contact_matrix is an NPZ file containing COO matrix components
    if isinstance(contact_matrix, np.lib.npyio.NpzFile):
        if all(key in contact_matrix for key in ['data', 'row', 'col', 'shape']):
            data = contact_matrix['data']
            row = contact_matrix['row']
            col = contact_matrix['col']
            shape = tuple(contact_matrix['shape'])
            contact_matrix = coo_matrix((data, (row, col)), shape=shape)
        else:
            logger.error("The contact matrix file does not contain the expected COO matrix keys.")
            raise ValueError("The contact matrix file does not contain the expected COO matrix keys.")
    
    # Validate matrix shape
    matrix_shape = contact_matrix.shape
    if matrix_shape[0] != matrix_shape[1]:
        logger.error("The contact matrix is not square.")
        raise ValueError("The contact matrix is not square.")
    if matrix_shape[0] != num_contigs:
        logger.error(f"The contact matrix dimensions {matrix_shape} do not match the number of contigs.")
        raise ValueError(f"The contact matrix dimensions {matrix_shape} do not match the number of contigs.")
    
    # Validate 'Self Contact' column if present
    if 'Self Contact' in contig_data.columns:
        # Convert COO to dense format for diagonal validation
        diagonal_values = contact_matrix.diagonal()
        self_contact = contig_data['Self Contact'].dropna()
        if not np.allclose(self_contact, diagonal_values[:len(self_contact)]):
            logger.error("The 'Self Contact' column values do not match the diagonal of the contact matrix.")
            raise ValueError("The 'Self Contact' column values do not match the diagonal of the contact matrix.")
    
    return True

def validate_unnormalized_folder(folder):
    expected_files = ['contig_info_final.csv', 'raw_contact_matrix.npz']
    missing_files = [file for file in expected_files if file not in folder]
    if missing_files:
        logger.error(f"Missing files in unnormalized folder: {', '.join(missing_files)}")
        raise ValueError(f"Missing files in unnormalized folder: {', '.join(missing_files)}")
    return True

def validate_normalized_folder(folder):
    expected_files = ['bin_info_final.csv', 'contig_info_final.csv', 'contig_contact_matrix.npz', 'bin_contact_matrix.npz']
    missing_files = [file for file in expected_files if file not in folder]
    if missing_files:
        logger.error(f"Missing files in normalized folder: {', '.join(missing_files)}")
        raise ValueError(f"Missing files in normalized folder: {', '.join(missing_files)}")
    return True

def list_files_in_7z(decoded):
    with py7zr.SevenZipFile(io.BytesIO(decoded), mode='r') as z:
        file_list = z.getnames()
    return file_list

def adjust_taxonomy(row, taxonomy_columns, prefixes):
    last_non_blank = ""
    
    for tier in taxonomy_columns:
        row[tier] = str(row[tier]) if pd.notna(row[tier]) else ""

    if row['Type'] != 'unmapped':
        for tier in taxonomy_columns:
            if row[tier]:
                last_non_blank = row[tier]
            else:
                row[tier] = f"Unspecified {last_non_blank}"
    else:
        for tier in taxonomy_columns:
            row[tier] = "unmapped"

    if row['Type'] == 'phage':
        row['Domain'] = 'Virus'
        row['Phylum'] = 'Virus'
        row['Class'] = 'Virus'
        for tier in taxonomy_columns:
            row[tier] = row[tier] + '_v'
        row['Contig'] = row['Contig'] + "_v"
        row['Bin'] = row['Bin'] + "_v"

    if row['Type'] == 'plasmid':
        for tier in taxonomy_columns:
            row[tier] = row[tier] + '_p'
        row['Contig'] = row['Contig'] + "_p"
        row['Bin'] = row['Bin'] + "_p"

    for tier, prefix in prefixes.items():
        row[tier] = f"{prefix}{row[tier]}" if row[tier] else "N/A"

    return row

def process_data(contig_data, binning_data, taxonomy_data, contig_matrix, user_folder):
    try:
        logger.info("Starting data preparation...")
        
        if isinstance(contig_matrix, str):
            logger.error("contig_matrix is a string, which is unexpected. Please check the source.")
            raise ValueError("contact_matrix should be a sparse matrix, not a string.")
        
        # Ensure the contact_matrix is in the correct sparse format if it's not already
        if not isinstance(contig_matrix, coo_matrix):
            logger.error("contig_matrix is not a COO sparse matrix.")
            raise ValueError("contig_matrix must be a COO sparse matrix.")

        # Query plasmid classification for any available plasmid IDs
        logger.info("Querying plasmid IDs for classification...")
        plasmid_ids = taxonomy_data['Plasmid ID'].dropna().unique().tolist()
        plasmid_classification_df = query_plasmid_id(plasmid_ids)

        # Ensure plasmid_classification_df has the expected columns
        expected_columns = [
            'NUCCORE_ACC', 
            'TAXONOMY_superkingdom', 
            'TAXONOMY_phylum', 
            'TAXONOMY_class', 
            'TAXONOMY_order', 
            'TAXONOMY_family', 
            'TAXONOMY_genus', 
            'TAXONOMY_species'
        ]

        # Check if the returned dataframe contains all expected columns
        if not all(col in plasmid_classification_df.columns for col in expected_columns):
            logger.error("The plasmid classification data does not contain the expected columns.")
            raise ValueError("The plasmid classification data does not contain the expected columns.")

        # Rename columns to match internal structure
        plasmid_classification_df.rename(columns={
            'NUCCORE_ACC': 'Plasmid ID',
            'TAXONOMY_superkingdom': 'Kingdom',
            'TAXONOMY_phylum': 'Phylum',
            'TAXONOMY_class': 'Class',
            'TAXONOMY_order': 'Order',
            'TAXONOMY_family': 'Family',
            'TAXONOMY_genus': 'Genus',
            'TAXONOMY_species': 'Species'
        }, inplace=True)

        # Define prefixes for taxonomy tiers
        prefixes = {
            'Domain': 'd_',
            'Kingdom': 'k_',
            'Phylum': 'p_',
            'Class': 'c_',
            'Order': 'o_',
            'Family': 'f_',
            'Genus': 'g_',
            'Species': 's_'
        }

        # Replace certain text in the classification dataframe
        plasmid_classification_df = plasmid_classification_df.replace(r"\s*\(.*\)", "", regex=True)
        plasmid_classification_df['Domain'] = plasmid_classification_df['Kingdom']

        # Merge plasmid classification with taxonomy data
        taxonomy_columns = ['Domain', 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
        taxonomy_data = taxonomy_data.merge(
            plasmid_classification_df[['Plasmid ID'] + taxonomy_columns],
            on='Plasmid ID',
            how='left',
            suffixes=('', '_new')
        )

        # Fill taxonomy columns with new classification where available
        for column in taxonomy_columns:
            taxonomy_data[column] = taxonomy_data[column + '_new'].combine_first(taxonomy_data[column])

        # Drop unnecessary columns after merge
        taxonomy_data = taxonomy_data.drop(columns=['Plasmid ID'] + [col + '_new' for col in taxonomy_columns])

        # Merge contig, binning, and taxonomy data
        combined_data = pd.merge(contig_data, binning_data, on="Contig", how="left")
        combined_data = pd.merge(combined_data, taxonomy_data, on="Bin", how="left")

        # Apply taxonomy adjustments
        combined_data = combined_data.apply(lambda row: adjust_taxonomy(row, taxonomy_columns, prefixes), axis=1)

        # Fill missing bins with 'Unbinned MAG'
        combined_data['Bin'] = combined_data['Bin'].fillna('Unbinned MAG')
        
        # Set the 'Signal' column in combined_data using the diagonal values from the contact matrix
        diagonal_values = contig_matrix.diagonal()
        combined_data['Signal'] = diagonal_values

        # Return the processed combined data directly
        logger.info("Data processed successfully.")
        return combined_data

    except Exception as e:
        logger.error(f"Error during data preparation: {e}; no preview will be generated.")
        return None

def preprocess_normalization(user_folder, assets_folder='output'):
    try:
        logger.info("Starting data preprocessing...")
        
        # Locate the folder path for the data preparation output
        folder_path = os.path.join('assets', assets_folder, user_folder)
        logger.info(f"Folder path for data preparation output: {folder_path}")

        # Define paths for the files within the folder
        contig_info_path = os.path.join(folder_path, 'contig_info_final.csv')
        contact_matrix_path = os.path.join(folder_path, 'raw_contact_matrix.npz')

        # Read the contig information file as a pandas DataFrame
        logger.info(f"Reading contig information file from: {contig_info_path}")
        contig_info = pd.read_csv(contig_info_path)

        # Load the contact matrix from .npz and reconstruct it as a sparse COO matrix
        logger.info(f"Loading contact matrix from: {contact_matrix_path}")
        contact_matrix_data = np.load(contact_matrix_path)
        data = contact_matrix_data['data']
        row = contact_matrix_data['row']
        col = contact_matrix_data['col']
        shape = tuple(contact_matrix_data['shape'])
        contact_matrix = coo_matrix((data, (row, col)), shape=shape)

        logger.info("Data preprocessing completed successfully.")
        return contig_info, contact_matrix

    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        return None, None

def run_normalization(method, contig_df, contact_matrix, epsilon=1, threshold=5, max_iter=1000, tolerance=0.000001):
    # Ensure contact_matrix is in coo format for consistency across methods
    contact_matrix = contact_matrix.tocoo()
    
    def safe_square(array):
        array = np.clip(array, -1e10, 1e10)  # Clip large values for safety
        return array ** 2
    
    def safe_divide(array, divisor):
        divisor = np.where(divisor == 0, epsilon, divisor)  # Add epsilon to avoid zero division
        return array / divisor

    def standardize(array):
        std = np.std(array)
        return np.zeros_like(array) if std == 0 else (array - np.mean(array)) / std

    def denoise(matrix, threshold):
        matrix = matrix.tocoo()
        threshold_value = np.percentile(matrix.data, threshold)
        mask = matrix.data > threshold_value
        return coo_matrix((matrix.data[mask], (matrix.row[mask], matrix.col[mask])), shape=matrix.shape)

    def _bisto_seq(m, max_iter, tol):
        m = m.tocoo()
        m.setdiag(np.where(m.diagonal() == 0, 1, m.diagonal()))  # Ensure no zeros on diagonal
        m = m.tocsr()
        
        n = m.shape[0]
        e = np.ones(n)
        x = e.copy()
        
        for i in range(max_iter):
            # Clip x to avoid overflow in the square operation
            x = np.clip(x, -1e10, 1e10)
            
            # Update x_new with added epsilon to avoid division by zero
            x_new = 1 / (m @ (x ** 2 + epsilon))
            
            # Check convergence with clipped difference for stability
            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                break
            
            # Handle NaNs and infs in x_new after each update
            x = np.nan_to_num(x_new, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Construct the bistochastic matrix using x
        X = spdiags(x, 0, n, n, format='csr')
        return X.T @ m @ X

    try:
        if method == 'Raw':
            logger.info("Running Raw normalization.")
            return denoise(contact_matrix, threshold)

        elif method == 'normCC':
            logger.info("Running normCC normalization.")
            signal = contact_matrix.max(axis=1).toarray().ravel()
            coverage = contig_df['Coverage'].values
            contact_matrix.setdiag(0)

            df = contig_df.copy()
            df['Coverage'] = coverage
            df['signal'] = signal

            logger.info("Performing log transformations for normCC.")
            df['log_site'] = np.log(df['Restriction sites'] + epsilon)
            df['log_len'] = np.log(df['Length'])
            df['log_coverage'] = np.log(df['Coverage'] + epsilon)

            exog = df[['log_site', 'log_len', 'log_coverage']]
            exog = sm.add_constant(exog)
            endog = df['signal']
            glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=1))
            res = glm_nb.fit()

            expected_signal = np.exp(np.dot(exog, res.params))
            scal = np.max(expected_signal)

            normalized_data = [scal * v / np.sqrt(expected_signal[i] * expected_signal[j])
                               for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data)]

            normalized_matrix = coo_matrix((normalized_data, (contact_matrix.row, contact_matrix.col)),
                                           shape=contact_matrix.shape)
            return denoise(normalized_matrix, threshold)

        elif method == 'HiCzin':
            logger.info("Running HiCzin normalization.")
            contact_matrix.setdiag(0)
            coverage = contig_df['Coverage'].replace(0, epsilon).values

            map_x = contact_matrix.row
            map_y = contact_matrix.col
            map_data = contact_matrix.data
            index = map_x < map_y
            map_x, map_y, map_data = map_x[index], map_y[index], map_data[index]

            sample_site = standardize(np.log(contig_df['Restriction sites'][map_x] * contig_df['Restriction sites'][map_y]))
            sample_len = standardize(np.log(contig_df['Length'][map_x] * contig_df['Length'][map_y]))
            sample_cov = standardize(np.log(coverage[map_x] * coverage[map_y]))

            data_hiczin = pd.DataFrame({
                'sample_site': sample_site,
                'sample_len': sample_len,
                'sample_cov': sample_cov,
                'sampleCon': map_data
            })

            exog = data_hiczin[['sample_site', 'sample_len', 'sample_cov']]
            exog = sm.add_constant(exog)
            endog = data_hiczin['sampleCon']

            glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=1))
            res = glm_nb.fit()

            expected_signal = np.exp(np.dot(exog, res.params))
            normalized_data = map_data / expected_signal

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (map_x, map_y)), shape=contact_matrix.shape
            )
            normalized_contact_matrix += normalized_contact_matrix.transpose()

            return denoise(normalized_contact_matrix, threshold)

        elif method == 'bin3C':
            logger.info("Running bin3C normalization.")
            num_sites = contig_df['Restriction sites'].values + epsilon
            normalized_data = [v / (num_sites[i] * num_sites[j])
                               for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data)]

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (contact_matrix.row, contact_matrix.col)), shape=contact_matrix.shape
            )

            bistochastic_matrix = _bisto_seq(normalized_contact_matrix, max_iter, tolerance)
            return denoise(bistochastic_matrix, threshold)

        elif method == 'MetaTOR':
            logger.info("Running MetaTOR normalization.")
            signal = contact_matrix.diagonal() + epsilon
            normalized_data = [v / np.sqrt(signal[i] * signal[j])
                               for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data)]

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (contact_matrix.row, contact_matrix.col)), shape=contact_matrix.shape
            )

            return denoise(normalized_contact_matrix, threshold)

    except Exception as e:
        logger.error(f"Error during {method} normalization: {e}")
        return None

def get_contig_indexes(annotations, contig_information):
    num_threads = 4 * os.cpu_count()
    
    if isinstance(annotations, str):
        annotations = [annotations]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {}
        for annotation in annotations:
            futures[executor.submit(
                lambda ann: (ann, contig_information[contig_information['Bin'] == ann].index.tolist()),
                annotation)] = annotation
        
        contig_indexes = {}

        for future in futures:
            annotation = futures[future]
            try:
                annotation, indexes = future.result()
                contig_indexes[annotation] = indexes
            except Exception as e:
                print(f'Error fetching contig indexes for annotation: {annotation}, error: {e}')
        
    if len(contig_indexes) == 1:
        return list(contig_indexes.values())[0]

    return contig_indexes

def generating_bin_information(contig_info, contact_matrix, remove_unmapped_contigs=False, remove_host_host=False):
    # Ensure dense matrix for processing
    dense_matrix = contact_matrix.toarray()

    # Handle unmapped contigs
    if remove_unmapped_contigs:
        unmapped_contigs = contig_info[contig_info['Type'] == "unmapped"].index.tolist()
        contig_info = contig_info.drop(unmapped_contigs).reset_index(drop=True)
        
        # Mask for rows/columns to keep
        keep_mask = np.ones(dense_matrix.shape[0], dtype=bool)
        keep_mask[unmapped_contigs] = False
        contig_contact_matrix = dense_matrix[keep_mask, :][:, keep_mask]
    else:
        contig_contact_matrix = dense_matrix.copy()

    # Aggregate bin data
    bin_data = contig_info.groupby('Bin').agg({
        'Contig': lambda x: ', '.join(x),
        'Restriction sites': 'sum',
        'Length': 'sum',
        'Coverage': 'sum',
        'Self Contact': 'sum',
        'Type': lambda x: x.mode()[0],
        'Domain': lambda x: x.mode()[0],
        'Kingdom': lambda x: x.mode()[0],
        'Phylum': lambda x: x.mode()[0],
        'Class': lambda x: x.mode()[0],
        'Order': lambda x: x.mode()[0],
        'Family': lambda x: x.mode()[0],
        'Genus': lambda x: x.mode()[0],
        'Species': lambda x: x.mode()[0]
    }).reset_index()

    unique_annotations = bin_data['Bin']
    contig_indexes_dict = get_contig_indexes(unique_annotations, contig_info)

    host_annotations = bin_data[bin_data['Type'] == 'chromosome']['Bin'].tolist()
    non_host_annotations = bin_data[~bin_data['Type'].isin(['chromosome'])]['Bin'].tolist()

    # Create the bin contact matrix
    bin_contact_matrix = pd.DataFrame(0.0, index=unique_annotations, columns=unique_annotations)

    if remove_host_host:
        # Step 1: Process interactions between non-host annotations
        for annotation_i in tqdm(non_host_annotations, desc="Processing non-host to non-host interactions"):
            for annotation_j in non_host_annotations:
                indexes_i = contig_indexes_dict[annotation_i]
                indexes_j = contig_indexes_dict[annotation_j]
                
                # Extract the submatrix and sum values
                sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
                bin_contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()

        # Step 2: Add interactions between host and non-host annotations
        for annotation_i in tqdm(host_annotations, desc="Processing host to non-host interactions"):
            for annotation_j in non_host_annotations:
                indexes_i = contig_indexes_dict[annotation_i]
                indexes_j = contig_indexes_dict[annotation_j]
                
                # Extract submatrix and sum values
                sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
                bin_contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()
                bin_contact_matrix.at[annotation_j, annotation_i] = sub_matrix.sum()
        
        # Step 3: Remove host-host interactions
        host_indexes = []
        for host_annotation in host_annotations:
            host_indexes.extend(contig_indexes_dict[host_annotation])
        
        # Mask to exclude host-host contacts
        host_mask = np.ones(dense_matrix.shape[0], dtype=bool)
        host_mask[host_indexes] = False
        
        # Remove host-host interactions
        contig_contact_matrix = dense_matrix[host_mask, :][:, host_mask]

    else:
        # Process all interactions including host-host
        for annotation_i in tqdm(unique_annotations, desc="Processing contact matrix"):
            for annotation_j in unique_annotations:
                indexes_i = contig_indexes_dict[annotation_i]
                indexes_j = contig_indexes_dict[annotation_j]
                
                # Extract submatrix and sum values
                sub_matrix = dense_matrix[np.ix_(indexes_i, indexes_j)]
                bin_contact_matrix.at[annotation_i, annotation_j] = sub_matrix.sum()

    # Convert to COO sparse matrices for storage
    bin_contact_matrix = coo_matrix(bin_contact_matrix)
    contig_contact_matrix = coo_matrix(contig_contact_matrix)

    return bin_data, contig_info, bin_contact_matrix, contig_contact_matrix