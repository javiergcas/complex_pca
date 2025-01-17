import argparse
import nibabel as nb
import numpy as np
from scipy.stats import zscore
from utils.load_write import load_file, write_out
import os #CW added


def run_zscore(input_path, output_path, mask_path):
    print('++ [run_zscore]: Entering Run Recon...')
    print(' +              Path to mask file   = %s' % mask_path)
    print(' +              Path to input file  = %s' % input_path)
    print(' +              Path to output file = %s' % output_path)
    # Load Mask
    print('++ [run_zscore]: Load mask...')
    mask        = nb.load(mask_path)
    mask_bin    = mask.get_fdata() > 0
    
    # Load input data
    print('++ [run_zscore]: Load input data...')
    data,header = load_file(input_path,'nifti',mask_bin,False,None,None,None,True)
    data_n      = data.shape[0]
    
    # Z-score the data across the time dimension
    print('++ [run_zscore]: Z-score data across the time dimension...')
    data        = zscore(data, nan_policy='omit')
    data        = np.nan_to_num(data)

    # Write output dataset
    print('++ [run_zscore]: Write output dataset...')
    write_out(data,mask,header,'nifti',output_path)

    print('++ [run_zscore]: Program ends')
    
if __name__ == '__main__':
    """Create Z-scored version of input data (micmiking CPCA program)"""
    parser = argparse.ArgumentParser(description='Create Z-scored version of the data')
    parser.add_argument('-i', '--input',
                        help='<Required> path to input nifti file',
                        required=True,
                        type=str)
    parser.add_argument('-o','--output',
                        help='<Required> path to output nifti file',
                        required=True,
                        type=str)
    parser.add_argument('-m', '--mask',
                        help='path to mask in nifti format',
                        default=None,
                        required=True,
                        type=str)
    args_dict = vars(parser.parse_args())
    os.environ["MKL_INTERFACE_LAYER"] = "ILP64" #CW added
    run_zscore(args_dict['input'], args_dict['output'], args_dict['mask'])
