import os
import io
import numpy as np
import base64
from scipy.sparse import csr_matrix
import requests
import pandas
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def save_file_to_user_folder(contents, filename, user_folder, folder_name='output'):
    """
    Saves contents to a user-specific folder within the assets directory.

    Parameters:
    - contents: The data to save as base64 content string.
    - filename: The name of the file to save, including the extension (.csv, .npz, .7z).
    - user_folder: Unique identifier for the user's folder.
    - folder_name: Folder within assets to save the files (default is 'output').

    Returns:
    - file_path: Path to the saved file.
    """
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

        # Save as NPZ file
        elif filename.endswith('.npz'):
            npz_file = np.load(io.BytesIO(decoded))
            if all(key in npz_file for key in ['data', 'indices', 'indptr', 'shape']):
                csr = csr_matrix((npz_file['data'], npz_file['indices'], npz_file['indptr']), shape=tuple(npz_file['shape']))
                np.savez_compressed(file_path, data=csr.data, indices=csr.indices, indptr=csr.indptr, shape=csr.shape)
                print(f"NPZ file saved: {file_path}")
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


def query_plasmid_id(ids, fasta=False):
    """
    Processes the plasmid IDs and returns their information stored in PLSDB as .tsv and (optional) .fasta file.
    :param ids: List of plasmid NCBI sequence accession ids
    :param fasta: When set to True fasta file is generated
    :return: Pandas dataframe containing plasmid info from PLSDB plus fasta sequence if fasta=True
    """

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
    logger = logging.getLogger('plsdbapi_logger')
    logger.setLevel(logging.INFO)
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
    df1 = pandas.DataFrame(data=found)
    df2 = pandas.DataFrame(data=notfound)
    df = pandas.concat([df1, df2], ignore_index=True, sort=False)

    return df