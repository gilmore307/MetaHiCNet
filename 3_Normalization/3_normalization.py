import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz, csr_matrix, coo_matrix, spdiags
import statsmodels.api as sm
import gc
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def standardize(array: np.ndarray) -> np.ndarray:
    """
    Standardizes the input array.
    
    Parameters:
        array (np.ndarray): Input array to standardize.
    
    Returns:
        np.ndarray: Standardized array or zero array if standard deviation is 0.
    """
    std = np.std(array)
    return np.zeros_like(array) if std == 0 else (array - np.mean(array)) / std

class Normalization:
    def __init__(self, epsilon=1, max_iter=1000):
        self.epsilon = epsilon
        self.max_iter = max_iter

    def preprocess(self, contig_file: str, contact_matrix_file: str, output_path: str, min_len=1000, min_signal=2, thres=5):
        """
        Preprocesses the input data: loads the contig information and contact matrix, applies filters, and sets up paths.
        
        Parameters:
            contig_file (str): Path to contig information CSV file.
            contact_matrix_file (str): Path to contact matrix npz file.
            output_path (str): Directory to store output.
            min_len (int): Minimum contig length to consider.
            min_signal (int): Minimum signal threshold.
            thres (float): Threshold percentage for denoising (0-100).
        """
        self.min_len = min_len
        self.min_signal = min_signal
        self.output_path = output_path
        self.thres = thres

        # Load contig information
        names = ['contig_name', 'sites', 'length', 'coverage']
        self.contig_info = pd.read_csv(
            contig_file,
            header=None,
            names=names,
            dtype={'sites': float, 'length': float, 'coverage': float}
        )

        # Filter contigs based on min_len
        self.contig_info = self.contig_info[self.contig_info['length'] >= self.min_len].reset_index(drop=True)
        logging.info(f"Filtered contigs based on min_len: {self.min_len}")

        # Load contact matrix with error handling
        try:
            contact_matrix_full = load_npz(contact_matrix_file).tocoo()
            logging.info("Loaded contact matrix.")
        except FileNotFoundError:
            logging.error(f"Error: The file '{contact_matrix_file}' was not found.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error loading file '{contact_matrix_file}': {e}")
            sys.exit(1)

        self.contact_matrix = contact_matrix_full

        # Make output folder
        os.makedirs(self.output_path, exist_ok=True)
        logging.info(f"Output directory created at {self.output_path}")

    def raw(self):
        """
        Do not conduct any normalization, just denoise.
        """
        try:
            logging.info("Starting raw normalization.")
            contact_matrix_raw = self.contact_matrix.copy()
            self.denoise(contact_matrix_raw, 'raw')
            del contact_matrix_raw
            gc.collect()
        except Exception as e:
            logging.error(f"Error during raw normalization: {e}")

    def normcc(self):
        """
        Perform normCC normalization.
        """
        try:
            logging.info("Starting normCC normalization.")
            contact_matrix = self.contact_matrix.copy()
            covcc = contact_matrix.diagonal()
            contact_matrix.setdiag(0)
            signal = contact_matrix.max(axis=1).toarray().ravel()
            site = self.contig_info['sites'].values
            length = self.contig_info['length'].values

            logging.info("Running GLM for normCC normalization.")

            contig_info_normcc = pd.DataFrame({
                'site': site,
                'length': length,
                'covcc': covcc,
                'signal': signal
            })

            contig_info_normcc['sample_site'] = np.log(contig_info_normcc['site'] + self.epsilon)
            contig_info_normcc['sample_len'] = np.log(contig_info_normcc['length'])
            contig_info_normcc['sample_covcc'] = np.log(contig_info_normcc['covcc'] + self.epsilon)
            exog = contig_info_normcc[['sample_site', 'sample_len', 'sample_covcc']]
            endog = contig_info_normcc['signal']
            exog = sm.add_constant(exog)
            glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=1))
            res = glm_nb.fit(method="lbfgs")
            norm_result = res.params.values

            linear_predictor = np.dot(exog, norm_result)
            expected_signal = np.exp(linear_predictor)
            scal = np.max(expected_signal)

            # Normalize contact values
            normalized_data = []
            for i, j, v in zip(contact_matrix.row, contact_matrix.col, contact_matrix.data):
                mu_i = expected_signal[i]
                mu_j = expected_signal[j]
                normalized_value = scal * v / np.sqrt(mu_i * mu_j)
                normalized_data.append(normalized_value)

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (contact_matrix.row, contact_matrix.col)),
                shape=contact_matrix.shape
            )

            self.denoise(normalized_contact_matrix, 'normcc')
            del contact_matrix, normalized_contact_matrix
            gc.collect()

        except Exception as e:
            logging.error(f"Error during normCC normalization: {e}")
            return None

    def hiczin(self):
        """
        Perform HiCzin normalization.
        """
        try:
            logging.info("Starting HiCzin normalization.")
            contact_matrix = self.contact_matrix.copy()
            contact_matrix.setdiag(0)

            contig_info_hiczin = self.contig_info.copy()
            contig_info_hiczin['site'] = contig_info_hiczin['sites'] + self.epsilon

            if 'coverage' not in contig_info_hiczin.columns or contig_info_hiczin['coverage'].isnull().any():
                logging.warning("'coverage' column is missing or contains NaNs. Replacing with epsilon.")
                contig_info_hiczin['coverage'] = self.epsilon
            else:
                min_non_zero = contig_info_hiczin['coverage'][contig_info_hiczin['coverage'] > 0].min()
                contig_info_hiczin['coverage'] = contig_info_hiczin['coverage'].replace(0, min_non_zero)

            map_x = contact_matrix.row
            map_y = contact_matrix.col
            map_data = contact_matrix.data
            index = map_x < map_y
            map_x = map_x[index]
            map_y = map_y[index]
            map_data = map_data[index]

            sample_site = np.log(contig_info_hiczin['site'].values[map_x] * contig_info_hiczin['site'].values[map_y])
            sample_len = np.log(contig_info_hiczin['length'].values[map_x] * contig_info_hiczin['length'].values[map_y])
            sample_cov = np.log(contig_info_hiczin['coverage'].values[map_x] * contig_info_hiczin['coverage'].values[map_y])

            sample_site = standardize(sample_site)
            sample_len = standardize(sample_len)
            sample_cov = standardize(sample_cov)

            data_hiczin = pd.DataFrame({
                'sample_site': sample_site,
                'sample_len': sample_len,
                'sample_cov': sample_cov,
                'sampleCon': map_data
            })

            exog = data_hiczin[['sample_site', 'sample_len', 'sample_cov']]
            endog = data_hiczin['sampleCon']
            exog = sm.add_constant(exog)

            if np.isnan(exog.values).any() or np.isinf(exog.values).any():
                raise ValueError("exog contains inf or NaNs")

            glm_nb = sm.GLM(endog, exog, family=sm.families.NegativeBinomial(alpha=1))
            res = glm_nb.fit()
            norm_result = res.params.values
            linear_predictor = np.dot(exog, norm_result)
            expected_signal = np.exp(linear_predictor)
            normalized_data = map_data / expected_signal

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (map_x, map_y)),
                shape=contact_matrix.shape
            )
            normalized_contact_matrix = normalized_contact_matrix + normalized_contact_matrix.transpose()

            self.denoise(normalized_contact_matrix, 'hiczin')
            del contact_matrix, normalized_contact_matrix
            gc.collect()

        except Exception as e:
            logging.error(f"Error during HiCzin normalization: {e}")
            return None

    def bin3c(self):
        """
        Perform bin3C normalization.
        """
        try:
            logging.info("Starting bin3C normalization.")
            num_sites = self.contig_info['sites'].values
            num_sites = num_sites + self.epsilon

            normalized_data = []
            for i, j, v in zip(self.contact_matrix.row, self.contact_matrix.col, self.contact_matrix.data):
                s_i = num_sites[i]
                s_j = num_sites[j]
                norm_value = v / (s_i * s_j)
                normalized_data.append(norm_value)

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (self.contact_matrix.row, self.contact_matrix.col)),
                shape=self.contact_matrix.shape
            )

            bistochastic_matrix, _ = self._bisto_seq(normalized_contact_matrix, self.max_iter, 1e-6)

            self.denoise(bistochastic_matrix, 'bin3c')

        except Exception as e:
            logging.error(f"Error during bin3C normalization: {e}")
            return None

    def metator(self):
        """
        Perform MetaTOR normalization.
        """
        try:
            logging.info("Starting MetaTOR normalization.")
            signal = self.contact_matrix.tocsr().diagonal()
            signal = signal + self.epsilon

            normalized_data = []
            for i, j, v in zip(self.contact_matrix.row, self.contact_matrix.col, self.contact_matrix.data):
                cov_i = signal[i]
                cov_j = signal[j]
                norm_factor = np.sqrt(cov_i * cov_j)
                normalized_value = v / norm_factor
                normalized_data.append(normalized_value)

            normalized_contact_matrix = coo_matrix(
                (normalized_data, (self.contact_matrix.row, self.contact_matrix.col)),
                shape=self.contact_matrix.shape
            )

            self.denoise(normalized_contact_matrix, 'metator')

        except Exception as e:
            logging.error(f"Error during MetaTOR normalization: {e}")
            return None

    def denoise(self, _norm_matrix: coo_matrix, suffix: str):
        """
        Denoises the contact matrix by removing values below a certain threshold.
        
        Parameters:
            _norm_matrix (coo_matrix): The normalized contact matrix.
            suffix (str): A suffix to append to the output filename.
        """
        try:
            if not isinstance(_norm_matrix, coo_matrix):
                _norm_matrix = _norm_matrix.tocoo()

            denoised_matrix_file = os.path.join(self.output_path, f'denoised_contact_matrix_{suffix}.npz')

            if _norm_matrix.nnz == 0:
                logging.warning(f"The contact matrix '{suffix}' is empty. Skipping denoising.")
                denoised_contact_matrix = coo_matrix(_norm_matrix.shape)
                save_npz(denoised_matrix_file, denoised_contact_matrix)
                return denoised_contact_matrix

            if self.thres <= 0 or self.thres >= 100:
                logging.error("The threshold percentage must be between 0 and 100. Using default 5%.")
                self.thres = 5

            threshold = np.percentile(_norm_matrix.data, self.thres)
            mask = _norm_matrix.data > threshold

            if not np.any(mask):
                logging.warning(f"No contacts exceed the threshold in '{suffix}'. Denoised matrix will be empty.")
                denoised_contact_matrix = coo_matrix(_norm_matrix.shape)
            else:
                denoised_contact_matrix = coo_matrix(
                    (_norm_matrix.data[mask], (_norm_matrix.row[mask], _norm_matrix.col[mask])),
                    shape=_norm_matrix.shape
                )

            save_npz(denoised_matrix_file, denoised_contact_matrix)
            logging.info(f"Denoised normalized contact matrix '{suffix}' saved to {denoised_matrix_file}")

            return denoised_contact_matrix
        except Exception as e:
            logging.error(f"Error during denoising normalization for '{suffix}': {e}")
            return None
    
    def _bisto_seq(self, m: csr_matrix, max_iter: int, tol: float, x0: np.ndarray = None, delta: float = 0.1, Delta: float = 3):
        """
        Apply the bistochastic matrix balancing algorithm to normalize the matrix.
        
        Parameters:
            m (csr_matrix): The matrix to normalize.
            max_iter (int): Maximum number of iterations for convergence.
            tol (float): Tolerance for convergence.
            x0 (np.ndarray, optional): Initial scale vector. Defaults to ones.
            delta (float): Lower bound constraint on the scaling factors.
            Delta (float): Upper bound constraint on the scaling factors.
    
        Returns:
            csr_matrix: The bistochastically balanced matrix.
            np.ndarray: The final scaling factors used in the balancing process.
        """
        logging.info("Starting bistochastic matrix balancing.")
        
        # Copy the original matrix and ensure it's in the correct format
        _orig = m.copy()
        m = m.tolil()  # LIL format for efficient in-place modifications
        
        # Handle zero diagonals by replacing them with 1 to avoid exploding scale factors
        is_zero_diag = m.diagonal() == 0
        if np.any(is_zero_diag):
            logging.warning(f"Replacing {is_zero_diag.sum()} zero diagonals with ones.")
            m.setdiag(np.where(is_zero_diag, 1, m.diagonal()))  # Set diagonal to 1 where it was zero
        
        m = m.tocsr()  # Convert back to CSR for efficient matrix operations
        
        # Initialize variables
        n = m.shape[0]
        e = np.ones(n)
        x = x0.copy() if x0 is not None else e.copy()
        g = 0.9
        etamax = 0.1
        eta = etamax
        stop_tol = tol * 0.5
        rt = tol ** 2
        v = x * m.dot(x)
        rk = 1 - v
        rho_km1 = np.dot(rk, rk)
        rout = rho_km1
        rold = rout
        n_iter = 0
        
        # Begin iterative balancing
        while rout > rt and n_iter < max_iter:
            y = e.copy()
            inner_tol = max(rout * eta ** 2, rt)
            rho_km2 = None
    
            while rho_km1 > inner_tol:
                if rho_km2 is None:
                    Z = rk / v
                    p = Z
                    rho_km1 = np.dot(rk, Z)
                else:
                    beta = rho_km1 / rho_km2
                    p = Z + beta * p
                
                w = x * m.dot(x * p) + v * p
                alpha = rho_km1 / np.dot(p, w)
                ap = alpha * p
                ynew = y + ap
    
                # Enforce delta and Delta constraints
                if np.min(ynew) <= delta:
                    gamma = np.min((delta - y[ynew < delta]) / ap[ynew < delta])
                    y += gamma * ap
                    break
                if np.max(ynew) >= Delta:
                    gamma = np.min((Delta - y[ynew > Delta]) / ap[ynew > Delta])
                    y += gamma * ap
                    break
                
                y = ynew
                rk -= alpha * w
                rho_km2 = rho_km1
                Z = rk / v
                rho_km1 = np.dot(rk, Z)
    
                if np.any(np.isnan(x)):
                    raise RuntimeError("Scaling vector has developed NaN values!")
    
            x *= y
            v = x * m.dot(x)
            rk = 1 - v
            rho_km1 = np.dot(rk, rk)
            rout = rho_km1
            n_iter += 1
            rold = rout
            eta = max(min(g * (rout / rold), etamax), stop_tol / np.sqrt(rout))
    
        if n_iter >= max_iter:
            logging.warning(f"Matrix balancing did not converge after {max_iter} iterations.")
        else:
            logging.info(f"Converged after {n_iter} iterations.")
    
        # Return the bistochastically balanced matrix and the scaling vector
        X = spdiags(x, 0, n, n, format='csr')
        return X.T.dot(_orig.dot(X)), x


def preprocess_common_args(args, normalizer):
    """
    Common preprocessing for all normalization methods to avoid code repetition.
    """
    normalizer.preprocess(
        contig_file=args.contig_file,
        contact_matrix_file=args.contact_matrix_file,
        output_path=args.output_path,
        thres=args.thres
    )

def main():
    parser = argparse.ArgumentParser(description="Normalization tool for MetaHit pipeline.")
    subparsers = parser.add_subparsers(dest='command', help='Available normalization commands')

    # Raw normalization
    parser_raw = subparsers.add_parser('raw', help='Perform raw normalization')
    parser_raw.add_argument('--contig_file', '-c', required=True, help='Path to contig_info.csv')
    parser_raw.add_argument('--contact_matrix_file', '-m', required=True, help='Path to contact_matrix.npz')
    parser_raw.add_argument('--output_path', '-o', required=True, help='Output directory')
    parser_raw.add_argument('--min_len', type=int, default=1000, help='Minimum contig length')
    parser_raw.add_argument('--min_signal', type=int, default=2, help='Minimum signal')
    parser_raw.add_argument('--thres', type=float, default=5, help='Threshold percentage for denoising (0-100)')

    # normCC normalization
    parser_normcc = subparsers.add_parser('normcc', help='Perform normCC normalization')
    parser_normcc.add_argument('--contig_file', '-c', required=True, help='Path to contig_info.csv')
    parser_normcc.add_argument('--contact_matrix_file', '-m', required=True, help='Path to contact_matrix.npz')
    parser_normcc.add_argument('--output_path', '-o', required=True, help='Output directory')
    parser_normcc.add_argument('--thres', type=float, default=5, help='Threshold percentage for denoising (0-100)')


    # Subparser for HiCzin normalization
    parser_hiczin = subparsers.add_parser('hiczin', help='Perform HiCzin normalization')
    parser_hiczin.add_argument('--contig_file', required=True, help='Path to contig_info.csv')
    parser_hiczin.add_argument('--contact_matrix_file', required=True, help='Path to contact_matrix.npz')
    parser_hiczin.add_argument('--output_path', required=True, help='Output directory')
    parser_hiczin.add_argument('--epsilon', type=float, default=1, help='Epsilon value')
    parser_hiczin.add_argument('--thres', type=float, default=5, help='Threshold percentage for denoising (0-100)')

    # Subparser for bin3C normalization
    parser_bin3c = subparsers.add_parser('bin3c', help='Perform bin3C normalization')
    parser_bin3c.add_argument('--contig_file', required=True, help='Path to contig_info.csv')
    parser_bin3c.add_argument('--contact_matrix_file', required=True, help='Path to contact_matrix.npz')
    parser_bin3c.add_argument('--output_path', required=True, help='Output directory')
    parser_bin3c.add_argument('--epsilon', type=float, default=1, help='Epsilon value')
    parser_bin3c.add_argument('--max_iter', type=int, default=1000, help='Maximum iterations for Sinkhorn-Knopp')
    parser_bin3c.add_argument('--tol', type=float, default=1e-6, help='Tolerance for convergence')
    parser_bin3c.add_argument('--thres', type=float, default=5, help='Threshold percentage for denoising (0-100)')

    # Subparser for MetaTOR normalization
    parser_metator = subparsers.add_parser('metator', help='Perform MetaTOR normalization')
    parser_metator.add_argument('--contig_file', required=True, help='Path to contig_info.csv')
    parser_metator.add_argument('--contact_matrix_file', required=True, help='Path to contact_matrix.npz')
    parser_metator.add_argument('--output_path', required=True, help='Output directory')
    parser_metator.add_argument('--epsilon', type=float, default=1, help='Epsilon value')
    parser_metator.add_argument('--thres', type=float, default=5, help='Threshold percentage for denoising (0-100)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    normalizer = Normalization()

    if args.command == 'raw':
        preprocess_common_args(args, normalizer)
        normalizer.raw()
    elif args.command == 'normcc':
        preprocess_common_args(args, normalizer)
        normalizer.normcc()
    elif args.command == 'hiczin':
        preprocess_common_args(args, normalizer)
        normalizer.hiczin()
    elif args.command == 'bin3c':
        preprocess_common_args(args, normalizer)
        normalizer.bin3c()
    elif args.command == 'metator':
        preprocess_common_args(args, normalizer)
        normalizer.metator()

if __name__ == "__main__":
    main()
