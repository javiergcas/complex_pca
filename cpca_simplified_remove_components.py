import argparse
import numpy as np
import pandas as pd
import h5py
import os #CW added
from utils.load_write import load_data, write_modified_scans

import dask.array as da
from dask.distributed import Client

def run_recon(input_files, pca_path, n_comps_to_remove, mask_fp, normalize, verbose):
    print('++ [run_recon]: Entering Run Recon...')
    print(' +              Number of components to remove      = %s' % str(n_comps_to_remove))
    print(' +              Path to mask file                   = %s' % mask_fp)
    print(' +              Path to input/output file info file = %s' % input_files)
    print(' +              Path to pre-computed CPCA object    = %s' % pca_path)
    print(' +              Verbose?                            = %s' % str(verbose)) 
    print(' +              Normalize                           = %s' % str(normalize))
    # load dataset
    print(" + [run_recon]: Loading data into memory.....")
    func_data, mask, header, func_data_trs, _, _, out_removed_paths = load_data(input_files, 'nifti', mask_fp, normalize, False, None, None, None, verbose) 
    
    # Load CPCA objects
    print(" + [run_recon]: Load CPCA objects into memory with Dask....")
    dask_client = Client()
    print(dask_client)

    f = h5py.File(pca_path, "r")    
    pca_output = {key: da.from_array(f[key], chunks="auto") for key in f.keys()}
    
    print(' +              Loading s...', end='')
    s = pca_output['s'].compute()
    print(' [DONE] s.shape=%s' % str(s.shape))

    print(' +              Loading Va...', end='')
    Va = pca_output['Va'].compute()
    print(' [DONE] Va.shape=%s' % str(Va.shape))

    print(' +              Loading U...', end='')
    U = pca_output['U'].compute()
    print(' [DONE] U.shape=%s' % str(U.shape))

    # Set to Zero the components to be removed
    print('++ [run_recon]: Setting to zero the components to be removed')
    print(' +              s = %s' % str(s))
    s_modified = s.copy()
    s_modified[:n_comps_to_remove] = 0
    print(' +              s_modified = %s' % str(s_modified))

    # Reconstructing the data (in analytical form)
    print('++ [run_recon]: Computing the modified data...')
    func_data_modified = np.dot(U * s_modified, Va)
    # Extracting the real part of the reconstructed data
    print('++ [run_recon]: Extracting the real part of the modified data...')
    func_data_modified = np.real(func_data_modified)
    # Write reconstructed data to disk
    print('++ [run_recon]: Writing modified scans to disk...')
    write_modified_scans(func_data_modified,mask,header,'nifti',func_data_trs, out_removed_paths, verbose)
    print('++ [run_recon]: Program ends')
    
if __name__ == '__main__':
    """Reconstruct data using pre-existing CPCA decomposition after removal of a given number of components"""
    parser = argparse.ArgumentParser(description='Data recon from pre-computed CPCA')
    parser.add_argument('-i', '--input',
                        help='<Required> file path to .txt file containing the file paths '
                        'to individual fMRI scans in nifti, cifti or 2D matrices represented '
                        'in .txt. format. The 2D matrix should be observations in rows and '
                        'columns are voxel/ROI/vertices',
                        required=True,
                        type=str)
    parser.add_argument('-pca','--pca_path',
                        help='<Required> path to previously computed CPCA',
                        required=True,
                        type=str)
    parser.add_argument('-rn', '--n_comps_to_remove',
                        help='<Required> Number of components to remove when reconstructing the data',
                        required=False,
                        default=None,
                        type=int)
    parser.add_argument('-m', '--mask',
                        help='path to brain mask in nifti format. Only needed '
                        'if file_format="nifti"',
                        default=None,
                        required=False,
                        type=str)
    parser.add_argument('-v', '--verbose_off',
                        help='turn off printing',
                        action='store_false',
                        required=False)
    parser.add_argument('-norm', '--normalize',
                        help='Type of scan normalization before group '
                        'concatenation. It is recommend to z-score',
                        default='zscore',
                        required=False,
                        choices=['zscore', 'mean_center'],
                        type=str) 
    args_dict = vars(parser.parse_args())
    os.environ["MKL_INTERFACE_LAYER"] = "ILP64" #CW added
    run_recon(args_dict['input'], args_dict['pca_path'], args_dict['n_comps_to_remove'], args_dict['mask'],args_dict['normalize'],args_dict['verbose_off'])
