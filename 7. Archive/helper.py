import os
import io
import numpy as np
import py7zr
import pandas as pd
from scipy.sparse import csr_matrix

def save_file_to_user_folder(contents, filename, user_folder, folder_name='output'):
    """
    Saves contents to a user-specific folder within the assets directory.

    Parameters:
    - contents: The data to save. This could be a DataFrame (for CSV), 
                a sparse matrix (for NPZ), or a list of tuples for 7z (e.g., [('file.csv', df), ('matrix.npz', matrix)])
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
    
    # Save a CSV file if the filename ends with .csv
    if filename.endswith('.csv'):
        if isinstance(contents, pd.DataFrame):
            contents.to_csv(file_path, index=False)
        else:
            raise ValueError("For .csv files, contents must be a pandas DataFrame.")
    
    # Save a NPZ file if the filename ends with .npz
    elif filename.endswith('.npz'):
        if isinstance(contents, csr_matrix):
            np.savez_compressed(file_path, data=contents.data, indices=contents.indices, 
                                indptr=contents.indptr, shape=contents.shape)
        else:
            raise ValueError("For .npz files, contents must be a scipy.sparse csr_matrix.")
    
    # Save a 7z archive if the filename ends with .7z
    elif filename.endswith('.7z'):
        if isinstance(contents, list):
            with py7zr.SevenZipFile(file_path, 'w') as archive:
                for item_name, item_content in contents:
                    # Save DataFrame to a CSV in memory
                    if isinstance(item_content, pd.DataFrame):
                        csv_buffer = io.StringIO()
                        item_content.to_csv(csv_buffer, index=False)
                        archive.writestr(item_name, csv_buffer.getvalue().encode('utf-8'))
                    
                    # Save matrix to a NPZ in memory
                    elif isinstance(item_content, csr_matrix):
                        npz_buffer = io.BytesIO()
                        np.savez_compressed(npz_buffer, data=item_content.data, indices=item_content.indices, 
                                            indptr=item_content.indptr, shape=item_content.shape)
                        archive.writestr(item_name, npz_buffer.getvalue())
                    
                    else:
                        raise ValueError("7z contents must be either pandas DataFrames or scipy.sparse csr_matrices.")
        else:
            raise ValueError("For .7z files, contents must be a list of tuples (filename, content).")
    
    else:
        raise ValueError("Unsupported file format. Only .csv, .npz, and .7z are supported.")
    
    return file_path
