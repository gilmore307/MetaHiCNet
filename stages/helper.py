import os
import io
import numpy as np
import base64
import pandas as pd
import logging
from scipy.sparse import coo_matrix, isspmatrix_coo
from joblib import Parallel, delayed
from io import StringIO
import pickle
import json

logger = logging.getLogger("app_logger")

def save_file_to_user_folder(contents, filename, user_folder, folder_name='output'):
    # Ensure the user folder exists
    user_folder_path = os.path.join(folder_name, user_folder)
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

def get_indexes(annotations, information_table, column):
    # Ensure annotations is a list
    if isinstance(annotations, str):
        annotations = [annotations]

    # Parallel computation using joblib
    def fetch_indexes(annotation):
        try:
            indexes = information_table[information_table[column] == annotation].index.tolist()
            return annotation, indexes
        except Exception as e:
            print(f'Error fetching contig indexes for annotation: {annotation}, error: {e}')
            return annotation, []

    # Execute in parallel
    results = Parallel(n_jobs=-1)(
        delayed(fetch_indexes)(annotation) for annotation in annotations
    )

    # Aggregate results
    contig_indexes = {annotation: indexes for annotation, indexes in results}

    return contig_indexes

def calculate_submatrix_sum(pair, contig_indexes_dict, matrix):
    annotation_i, annotation_j = pair
    indexes_i = contig_indexes_dict[annotation_i]
    indexes_j = contig_indexes_dict[annotation_j]
    sub_matrix = matrix[np.ix_(indexes_i, indexes_j)]
    return annotation_i, annotation_j, sub_matrix.sum()

def save_to_redis(key, data):  # ttl is set to 600 seconds (10 minutes) by default
    from app import r
    from app import SESSION_TTL

    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to JSON
        json_data = data.to_json(orient='split')
        r.set(key, json_data.encode('utf-8'), ex=SESSION_TTL)  # Set TTL for DataFrame
    elif isspmatrix_coo(data):  # Check if the data is a COO sparse matrix
        # Serialize sparse matrix using pickle
        binary_data = pickle.dumps(data)
        r.set(key, binary_data, ex=SESSION_TTL)  # Set TTL for COO matrix
    elif isinstance(data, np.ndarray):
        # Serialize NumPy array using pickle
        binary_data = pickle.dumps(data)
        r.set(key, binary_data, ex=SESSION_TTL)  # Set TTL for NumPy array
    elif isinstance(data, list) or isinstance(data, dict):
        # Convert list or dict to JSON
        json_data = json.dumps(data)
        r.set(key, json_data.encode('utf-8'), ex=SESSION_TTL)  # Set TTL for list or dict
    else:
        # Raise an error for unsupported types
        raise ValueError(f"Unsupported data type: {type(data)}")

def load_from_redis(key):
    from app import r
    data = r.get(key)

    if data is None:
        raise KeyError(f"No data found in Redis for key: {key}")

    # First, try loading as binary data with pickle (for NumPy arrays, sparse matrices, etc.)
    try:
        obj = pickle.loads(data)
        # If the object is a COO matrix, convert it to a dense array
        if isspmatrix_coo(obj):
            return obj.toarray()  # Convert COO matrix to dense array
        return obj  # Return as-is for other pickled objects
    except (pickle.UnpicklingError, TypeError):
        pass  # If binary loading fails, continue to JSON loading

    # Try interpreting the data as JSON (for DataFrames, lists, or dictionaries)
    try:
        decoded_data = data.decode('utf-8')
        try:
            # Attempt to load as DataFrame format JSON
            return pd.read_json(StringIO(decoded_data), orient='split')
        except ValueError:
            # If not a DataFrame, check if it's a simple list or dictionary
            parsed_data = json.loads(decoded_data)
            # Return as a DataFrame if the parsed data is a list of lists or dict
            if isinstance(parsed_data, list) or isinstance(parsed_data, dict):
                return parsed_data
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"Unable to load data from Redis for key: {key}, unknown format.")