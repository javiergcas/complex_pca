import argparse
import numpy as np
import pandas as pd
import fbpca
import warnings
import h5py
from numpy.linalg import pinv
from scipy.io import savemat
from scipy.signal import hilbert
from scipy.stats import zscore
from utils.cpca_reconstruction import cpca_recon
from utils.load_write import load_data, write_out, write_modified_scans
from xmca.tools.rotation import varimax, promax


# def save_dict_to_hdf5(filename, data_dict):
#     with h5py.File(filename, 'w') as h5file:
#         for key, value in data_dict.items():
#             h5file.create_dataset(key, data=value)

def save_dict_to_hdf5(filename, data_dict):
    with h5py.File(filename, 'w') as h5file:
        for key, value in data_dict.items():
            if isinstance(value, np.ndarray):
                # Check if the array contains objects
                if value.dtype == np.dtype('O'):
                    # Try to convert the object array to a numeric array if possible
                    try:
                        value = np.array(value, dtype=np.float64)
                    except ValueError:
                        print(f"Skipping key {key} - cannot convert to numeric.")
                        continue
                h5file.create_dataset(key, data=value)
            else:
                print(f"Skipping key {key} - not a numpy array.")

def hilbert_transform(input_data, verbose):
    # hilbert transform
    input_data = hilbert(input_data, axis=0)
    return input_data.conj()

def package_parameters(n_comps, mask_fp, file_format,
                       pca_type, rotate, normalize, bandpass, 
                       low_cut, high_cut, tr):
    # place input parameters into dictionary to write with results
    params = {
        'n_components': n_comps,
        'mask': mask_fp,
        'file_format': file_format,
        'pca_type': pca_type,
        'rotate': rotate,
        'normalize': normalize,
        'bandpass': bandpass,
        'lowcut': low_cut,
        'highcut': high_cut,
        'tr': tr
    }
    if mask_fp is None:
        params['mask'] = ''
    if rotate is None:
        params['rotate'] = ''
    if not bandpass:
        print('bandpass not activated --> setting all bandpass related inputs to empty')
        params['bandpass'] = ''
        params['lowcut'] = ''
        params['highcut'] = ''
    if tr is None:
        params['tr'] = ''
    return params


def pca(input_data, n_comps, pca_type, verbose, n_iter=10):
    # compute pca
    print('performing PCA/CPCA')
    # get number of observations
    n_samples  = input_data.shape[0]
    n_vertices = input_data.shape[1] 
    print(' number of samples         = %d' % n_samples)
    print(' number of vertices/voxels = %d' % n_vertices)
    #matrix_rank = np.linalg.matrix_rank(input_data)
    #print(' rank of input matrix = % s' % str(matrix_rank))
    # fbpca pca
    (U, s, Va) = fbpca.pca(input_data, k=n_comps, n_iter=n_iter)
    # calc explained variance
    # 1. Get eigs
    eigs = (s ** 2) / (n_samples-1)
    # 2. Compute Variance Explained from eigenvaluesa
    #    Assumptions:
    #    a) We are working with complex data --> we count the number of vertices twice
    #    b) Input were normalized timseries --> each vertex/voxel has a variance = 1
    if pca_type == 'complex':
       explained_variance_ = np.array([eig/(n_vertices*2) for eig in eigs])
    if pca_type == 'real':
       explained_variance_ = np.array([eig/(n_vertices) for eig in eigs])
    # Original Formulation
    # explained_variance_ = ((s ** 2) / (n_samples - 1)) / input_data.shape[1]
    total_var = explained_variance_.sum()
    
    # 3. Compute PC scores
    pc_scores = input_data @ Va.T
    # get loadings from eigenvectors
    loadings =  Va.T @ np.diag(s) 
    loadings /= np.sqrt(input_data.shape[0]-1)
    
    # 4. Package outputs
    output_dict = {'U': U,
                   's': s,
                   'Va': Va,
                   'loadings': loadings.T,
                   'exp_var': explained_variance_,
                   'eigs': eigs,
                   'pc_scores': pc_scores,
                   'n_samples': n_samples,
                   'n_positions': input_data.shape[1],
                   'total_var': total_var}
    return output_dict


def rotation(pca_output, data, rotation, verbose):
    print(f'applying {rotation} to PCA/CPCA loadings')
    # rotate PCA weights, if specified, and recompute pc scores
    if rotation == 'varimax':
        rotated_weights, r_mat = varimax(pca_output['loadings'].T)
        pca_output['r_mat'] = r_mat
    elif rotation == 'promax':
        rotated_weights, r_mat, phi_mat = promax(pca_output['loadings'].T)
        pca_output['r_mat'] = r_mat
        pca_output['phi_mat'] = phi_mat
    # https://stats.stackexchange.com/questions/59213/how-to-compute-varimax-rotated-principal-components-in-r
    # recompute pc scores
    projected_scores = data @ pinv(rotated_weights).T
    pca_output['loadings'] = rotated_weights.T
    pca_output['pc_scores'] = projected_scores
    return pca_output


def write_results(pca_output, pca_type, mask, file_format,
                  header, rotate, out_prefix, n_comps_to_save):
    # write out results of pca analysis
    # create output name if out_prefix is None
    if out_prefix is None:
        if pca_type == 'complex':
            out_prefix = f'cpca'
        elif pca_type == f'real':
            out_prefix = 'pca'
        if rotate is not None:
            out_prefix += f'_{rotate}'
        out_prefix += '_results'

    # Write variance explained
    out_df = None
    if 's_modified' in pca_output: 
       out_df = pd.DataFrame(np.vstack([pca_output['s'], pca_output['eigs'], pca_output['exp_var'], pca_output['s_modified']]).T,columns=['s','eigs','exp_var','s_modified'])
    else:
       out_df = pd.DataFrame(np.vstack([pca_output['s'], pca_output['eigs'], pca_output['exp_var']]).T,columns=['s','eigs','exp_var'])
    out_df.index.name = 'cpca_id'
    out_df_path = f'{out_prefix}_exp_var.txt'
    out_df.to_csv(out_df_path)
    print(" + [write_results]: Explained Variance Dataframe written to disk [%s]" % out_df_path)

    # get loadings
    loadings = pca_output['loadings'][0:n_comps_to_save:]
    print('=====================')
    print(loadings.shape)
    print(type(pca_output))
    print('=====================')
    # Write brain maps
    if file_format in ('nifti', 'cifti'):
        if pca_type == 'complex': 
            # if complex, write out real, imaginary comps, amplitude and angles
            write_out(
              np.abs(loadings), mask, header, 
              file_format, f'{out_prefix}_magnitude'
            )
            write_out(
              np.real(loadings), mask, header, 
              file_format, f'{out_prefix}_real'
            )
            write_out(
              np.imag(loadings), mask, header, 
              file_format, f'{out_prefix}_imag'
            )
            write_out(
              np.angle(loadings), mask, header, 
              file_format, f'{out_prefix}_phase'
            )
        elif pca_type == 'real':
            write_out(
              loadings, mask, header, file_format, out_prefix
            )
    # write out pca results dictionary to .mat file
    print(" + Starting to save pca_output as h5 object",end=' ')
    save_dict_to_hdf5(f'{out_prefix}.h5', pca_output)
    #savemat(f'{out_prefix}.mat', pca_output)
    print("[DONE]")

def run_cpca(input_files, n_comps, mask_fp, file_format, out_prefix, 
             pca_type, rotate, recon, normalize, bandpass, 
             low_cut, high_cut, tr, n_bins, verbose,recon_data,n_comps_to_remove, n_comps_to_recon,save_pca_out):
    print('++ Entering Run cpca...')
    print(' + number of components to recon   = %s' % str(n_comps_to_recon))
    print(' + number of components to compute = %s' % str(n_comps))
    print(' + number of components to remove  = %s' % str(n_comps_to_remove))
    print(' + recon data after component removal? %s' % str(recon_data))
    print(' + recon components separately? %s' % str(recon))
    print(' + bandpass = %s' % str(bandpass))
    print(' + verbose  = %s' % str(verbose))  
    # load dataset
    print("++ Loading data into memory.....")
    func_data, mask, header, func_data_trs, input_paths, out_asis_paths, out_removed_paths = load_data(
        input_files, file_format, mask_fp, normalize, 
        bandpass, low_cut, high_cut, tr, verbose
    ) 
    # if pca_type is complex, compute hilbert transform
    if pca_type == 'complex':
        print(' + Applying Hilbert Transform ...')
        func_data = hilbert_transform(func_data, verbose)

    # if n_comps not provided, set it to maximum possible
    if n_comps is None:
        n_comps = np.min(func_data.shape)
        print(f' + Automatically setting n_comps = {n_comps}')
    # compute pca
    pca_output = pca(func_data, n_comps, pca_type, verbose)

    # rotate pca weights, if specified
    if rotate is not None:
        print(' + Applying rotation ...')
        pca_output = rotation(pca_output, func_data, rotate, verbose)

    # free memory
    print(' + Freeing memory by deleting the func_data variable')
    del func_data

    # if cpca, and recon_data = True, create reconstructed data after removal of complex PC components
    # ================================================================================================
    if recon_data & (pca_type == 'complex') & (n_comps_to_remove is not None):
        if verbose:
            print(f'performing data reconstruction after removal of {n_comps_to_remove} components')
        # Nulling the first n_comps_to_remove components
        pca_output['s_modified'] = pca_output['s'].copy()
        pca_output['s_modified'][:n_comps_to_remove] = 0
        # Reconstructing the data (in analytical form)
        func_data_modified = np.dot(pca_output['U'] * pca_output['s_modified'], pca_output['Va'])
        func_data_allcomps = np.dot(pca_output['U'] * pca_output['s'], pca_output['Va'])
        # Extracting the real part of the reconstructed data
        func_data_modified = np.real(func_data_modified)
        func_data_allcomps = np.real(func_data_allcomps)
        # Write reconstructed data to disk
        write_modified_scans(func_data_modified,mask,header,file_format,func_data_trs, out_removed_paths, verbose)
        write_modified_scans(func_data_allcomps,mask,header,file_format,func_data_trs, out_asis_paths, verbose)
        # free memory
        print(' + Freeing memory by deleting the func_data_modified variable')
        del func_data_modified
        del func_data_allcomps
    
    # if cpca, and recon=True, create reconstructed time courses of complex PC
    # ========================================================================
    if recon & (pca_type == 'complex'):
        if verbose:
            print('performing CPCA component time series reconstruction')
        if n_comps_to_recon > 10:
            warnings.warn(
              'the # of components estimated is large, CPCA reconstruction '
              'will create a separate file for each component. This may take '
              'a while.'
          )
        del func_data # free up memory
        cpca_recon(pca_output, rotate, file_format,
                   mask, header, out_prefix, n_bins, n_comps_to_recon)
    elif recon & (pca_type == 'real'):
        warnings.warn('Time series reconstruction only available for CPCA')

    # put input parameters into PCA results dicitonary
    pca_output['params'] = package_parameters(
        n_comps, mask_fp, file_format, pca_type, rotate, 
        normalize, bandpass, low_cut, high_cut, tr
    )
    # write out results
    if verbose:
        print('writing out results')
    if save_pca_out:
        print(' writing pca_output dictorionary to disk')
        write_results(pca_output, pca_type, mask, file_format, 
                  header, rotate, out_prefix, n_comps_to_recon)


if __name__ == '__main__':
    """Run complex or standard principal component analysis"""
    parser = argparse.ArgumentParser(description='Run CPCA or PCA analysis')
    parser.add_argument('-i', '--input',
                        help='<Required> file path to .txt file containing the file paths '
                        'to individual fMRI scans in nifti, cifti or 2D matrices represented '
                        'in .txt. format. The 2D matrix should be observations in rows and '
                        'columns are voxel/ROI/vertices',
                        required=True,
                        type=str)
    parser.add_argument('-n', '--n_comps',
                        help='<Required> Number of components from PCA',
                        required=False,
                        default=None,
                        type=int)
    parser.add_argument('-n_comps_to_recon', '--n_comps_to_recon',
                        help='<Required> Number of components from PCA to reconstruct (one at a time)',
                        required=False,
                        default=None,
                        type=int)
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
    parser.add_argument('-f', '--file_format',
                        help='the file format of the individual fMRI scans '
                        'specified in input',
                        required=False,
                        default='nifti',
                        choices=['nifti', 'cifti', 'txt'],
                        type=str)
    parser.add_argument('-o', '--output_prefix',
                        help='the output file name. Default will be to save '
                        'to current working directory with standard name',
                        required=False,
                        default=None,
                        type=str)
    parser.add_argument('-t', '--pca_type',
                        help='Calculate complex or real PCA',
                        default='complex',
                        choices=['real', 'complex'],
                        type=str)
    parser.add_argument('-r', '--rotate',
                        help='Whether to rotate pca weights',
                        default=None,
                        required=False,
                        choices=['varimax', 'promax'],
                        type=str)
    parser.add_argument('-recon', '--recon',
                        help='Whether to reconstruct time courses from '
                        'complex PCA',
                        action='store_true',
                        required=False)
    parser.add_argument('-recon_data', '--recon_data',
                        help='Whether to reconstruct the data after removal of compoments',
                        action='store_true',
                        required=False)
    parser.add_argument('-save_pca_out', '--save_pca_out',
                        help='Save the PCA dictionary',
                        action='store_true',
                        required=False)
    parser.add_argument('-norm', '--normalize',
                        help='Type of scan normalization before group '
                        'concatenation. It is recommend to z-score',
                        default='zscore',
                        required=False,
                        choices=['zscore', 'mean_center'],
                        type=str)
    parser.add_argument('-b', '--bandpass_filter',
                        help='Whether to bandpass filter time course w/ '
                        'a butterworth filter',
                        action='store_true',
                        required=False)
    parser.add_argument('-f_low', '--bandpass_filter_low',
                        help='Low cut frequency for bandpass filter in Hz',
                        required=False,
                        default=0.01,
                        type=float
                        )
    parser.add_argument('-f_high', '--bandpass_filter_high',
                        help='High cut frequency for bandpass filter in Hz',
                        required=False,
                        default=0.1,
                        type=float
                        )
    parser.add_argument('-tr', '--sampling_unit',
                        help='The sampling unit of the signal - i.e. the TR',
                        required=False,
                        default=None,
                        type=float
                        )
    parser.add_argument('-n_bins', '--n_recon_bins',
                        help='Number of phase bins for reconstruction of CPCA '
                        'components. Higher number results in finer temporal '
                        'resolution',
                        required=False,
                        default=30,
                        type=int
                        )
    parser.add_argument('-v', '--verbose_off',
                        help='turn off printing',
                        action='store_false',
                        required=False)
    

    args_dict = vars(parser.parse_args())
    run_cpca(args_dict['input'], args_dict['n_comps'], args_dict['mask'],
            args_dict['file_format'], args_dict['output_prefix'], 
            args_dict['pca_type'], args_dict['rotate'], 
            args_dict['recon'], args_dict['normalize'], 
            args_dict['bandpass_filter'], args_dict['bandpass_filter_low'],
            args_dict['bandpass_filter_high'], args_dict['sampling_unit'],
            args_dict['n_recon_bins'], args_dict['verbose_off'],args_dict['recon_data'],args_dict['n_comps_to_remove'],
            args_dict['n_comps_to_recon'],args_dict['save_pca_out'])
