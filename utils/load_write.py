import nibabel as nb 
import numpy as np
import os


from nibabel.filebasedimages import ImageFileError
from scipy.io import loadmat 
from scipy.stats import zscore
from scipy.signal import butter, sosfiltfilt
import os.path as osp

def butter_bandpass(lowcut, highcut, fs, order=5):
    # create butterworth bandpass filter based on lowcut and highcut
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def convert_2d_nifti(mask_bin, nifti_data):
    # convert 4d nifti to 2d matrix using mask
    nonzero_indx = np.nonzero(mask_bin)
    nifti_2d = nifti_data[nonzero_indx]
    return nifti_2d.T


def initialize_matrix(fps, nscans, file_format, mask, verbose):
    # initialize empty matrix for faster concatenation of scans
    # if file format is nifti, get # of voxels in mask
    if file_format == 'nifti':
        n_ts = len(np.nonzero(mask)[0])
    # if file format is cifti, get # of vertices and voxels from first scan
    elif file_format == 'cifti':
        cifti = nb.load(fps[0])
        n_ts = cifti.shape[1]
    # if file format is text, get # of time courses from first scan
    elif file_format == 'txt':
        txt = np.loadtxt(fps[0])
        n_ts = txt.shape[1]
    # loop through file paths, load header and get # of observations
    # and add up across files to get total # observations
    n_t = 0
    for fp in fps:
        if file_format == 'nifti':
            nifti = nb.load(fp)
            n_t += nifti.header['dim'][4]
        elif file_format == 'cifti':
            cifti = nb.load(fp)
            n_t += cifti.shape[0]
        elif file_format == 'txt':
            n_t += sum(1 for _ in open(fp))
    if verbose:
        print(f'initializing matrix of size ({n_t}, {n_ts})')
    # initialize matrix with zeros
    matrix_init = np.zeros((n_t, n_ts))
    return matrix_init


def load_data(input_files, file_format, mask_fp, normalize, 
              bandpass, low_cut, high_cut, tr, verbose):
    """This function loads resting-state data and concatenates it in time.
    
    Inputs
    ======

    Outputs
    =======
    data: concatenated data
    mask: mask
    header: header
    data_trs: number of acquisitions per scan
    input_paths: path to input files
    out_asis_paths: path to output files (all components)
    out_removed_paths: path to output files (after removal of components)
    """
    # read file paths from input .txt file
    fps = read_input_file(input_files)
    input_paths, out_asis_paths, out_removed_paths = None, None, None
    # Separate input and ouput paths
    if isinstance(fps[0],list):
        print(' + [load_data]: both input and output paths were provided')
        if len(fps[0]) != 3:
            print(' ++ ERROR: Not enought entries per line in input file.')
            exit(1) 
        input_paths    = [i[0] for i in fps]
        out_asis_paths = [i[1] for i in fps]
        out_removed_paths = [i[2] for i in fps]
    else:
        input_paths  = fps
    del fps
    
    # master function for loading and concatenating functional scans
    # parameters check
    parameter_check(input_paths, file_format, tr, bandpass, mask_fp)
    
    # Pull file paths
    data, mask, header, data_trs = load_scans(
        input_paths, file_format, mask_fp, normalize, bandpass, 
        low_cut, high_cut, tr, verbose
    )

    return data, mask, header, data_trs, input_paths, out_asis_paths, out_removed_paths


def load_scans(fps, file_format, mask_fp, normalize, 
               bandpass, low_cut, high_cut, tr, verbose):
    # scan duration in number of trs
    scan_trs = []
    # get # of scans
    n_scans = len(fps)
    # if file_format = 'nifti', load mask
    if file_format == 'nifti':
        mask = nb.load(mask_fp)
        mask_bin = mask.get_fdata() > 0
    else:
        mask = None
        mask_bin = None

    # initialize group matrix with zeros
    group_data = initialize_matrix(fps, n_scans, file_format, 
                                   mask_bin, verbose) 
    print(f'loading and concatenating {n_scans} scans')
    if bandpass and verbose:
        print(
          f'bandpass filtering of signals between {low_cut} - {high_cut} Hz '
          ' will be performed'
        )
    # initialize counter
    indx=0
    # Loop through files and concatenate/append
    for fp in fps:
        # load file
        data, header = load_file(fp, file_format, mask_bin, 
                                 bandpass, low_cut, high_cut,
                                 tr, verbose)
        # get # observations
        data_n = data.shape[0]
        scan_trs.append(data_n)
        # Normalize data before concatenation
        if normalize == 'zscore':
            print(f' zscoring {fp}')
            data = zscore(data, nan_policy='omit')
        elif normalize == 'mean_center':
            print(f' mean centering {fp}')
            data = data - np.mean(data, axis=0)
        # fill nans w/ 0 in regions of poor functional scan coverage
        data = np.nan_to_num(data)
        # fill group matrix with subject data
        group_data[indx:(indx+data_n), :] = data
        # increase counter by # of observations in subject scan
        indx += data_n

    return group_data, mask, header, scan_trs


def load_file(fp, file_format, mask, bandpass, 
              low_cut, high_cut, tr, verbose):
    # Load file based on file format
    if file_format == 'nifti':
        nifti = nb.load(fp)
        nifti_data = nifti.get_fdata()
        data = convert_2d_nifti(mask, nifti_data)
        header = nifti.header
    elif file_format == 'cifti':
        cifti = nb.load(fp)
        data = cifti.get_fdata()
        header = cifti.header
    elif file_format == 'txt':
        data = np.loadtxt(fp)
        header = None 

    # if bandpass = True, bandpass all time courses by (low_cut - high_cut Hz)
    if bandpass:
        npad=1000 # pad 1000 samples
        fs = 1/tr #tr to sampling rate
        sos = butter_bandpass(low_cut, high_cut, fs)
        # Median padding to reduce edge effects
        data_pad = np.pad(data,[(npad, npad), (0, 0)], 'median')
        # backward and forward filtering
        data_filt = sosfiltfilt(sos, data_pad, axis=0)
        # Cut padding to original signal
        data = data_filt[npad:-npad, :]
    return data, header



def parameter_check(fps, file_format, tr, bandpass, mask):
    # check errors that may have not been caught by argparse
    # ensure a mask is supplied if file_format='nifi'
    if (mask is None) and (file_format == 'nifti'):
        raise Exception(
            'a mask .nii file must be supplied when file format is nifti'
        )
    # ensure that TR is provided if bandpass = True
    if bandpass and (tr is None):
        raise Exception(
            'the TR must be supplied if bandpass=True'
        )
    # test whether the files in the input match those specified in file format
    try:
        data_obj = nb.load(fps[0])
        # check whether it is nifti
        if isinstance(data_obj, nb.nifti1.Nifti1Image):
            actual_format = 'nifti'
        elif isinstance(data_obj, nb.cifti2.cifti2.Cifti2Image):
            actual_format = 'cifti'
    except ImageFileError:
        actual_format = 'txt'

    if file_format != actual_format:
        raise Exception(
            f"""
            It looks like the file format specified: '{file_format}''
            does not match the file format of the input files: '{actual_format}'
            """
        )


def read_input_file(input_files):
    # load input .txt file specifying file paths
    with open(input_files, 'r') as file:
        fps = [line.rstrip() for line in file]
        # remove extra lines, if any
        fps = [line for line in fps if len(line)>0]
    # Only separate into input output if there is more than one entry per line
    print(' + [read_input_file]: Number of entries in input file: %d lines' % (len(fps)))
    if '\t' in fps[0]:
        fps = [item.split('\t') for item in fps]
        print(' + [read_input_file]: Number of paths per input: %d paths' % (len(fps[0])))
    elif ' ' in fps[0]:
        fps = [item.split(' ') for item in fps]
        print(' + [read_input_file]: Number of paths per input: %d paths' % (len(fps[0])))
    return fps

def write_out(data, mask, header, file_format, out_prefix):
    # write out brain maps to nifti or cifti
    if file_format == 'nifti':
        mask_bin = mask.get_fdata() > 0
        # get 2d matrix into 4d nifti space
        nifti_4d = np.zeros(mask.shape + (data.shape[0],), 
                            dtype=data.dtype)
        nifti_4d[mask_bin, :] = data.T
        # write out brain w/ nibabel and mask affine
        nifti_out = nb.Nifti2Image(nifti_4d, mask.affine)
        if (not out_prefix.endswith('.nii')) & (not out_prefix.endswith('.nii.gz')):
            out_prefix = out_prefix + '.nii'
        nb.save(nifti_out, out_prefix)
    elif file_format == 'cifti':
        # https://neurostars.org/t/alter-size-of-matrix-for-new-cifti-header-nibabel/20903/2
        # first get tr from Series axis
        tr = header.get_axis(0).step
        # Create new axes 0 to match new mat size and store orig axes 1
        ax_0 = nb.cifti2.SeriesAxis(0, tr, data.shape[0]) 
        ax_1 = header.get_axis(1)
        # Create new header and cifti object
        new_header = nb.cifti2.Cifti2Header.from_axes((ax_0, ax_1))
        cifti_out = nb.cifti2.Cifti2Image(data, new_header)
        # need to create a new header due to change in matrix shape
        nb.save(cifti_out, f'{out_prefix}.dtseries.nii')

def write_modified_scans(data,mask,header,file_format,data_trs,file_paths, verbose):
    ## # extracting output paths
    ## fps = read_input_file(file_paths)
    ## # Separate input and ouput paths
    ## if isinstance(fps[0],list):
    ##     fps_inputs  = [i[0] for i in fps]
    ##     fps_outputs = [i[1] for i in fps]
    ## else:
    ##     fps_inputs  = fps
    ##     fps_outputs = None
    ## del fps
    
    # initlializing tr index
    indx = 0
    n_files_to_write = len(file_paths)
    for i,out_path in enumerate(file_paths):
        ii         = i + 1
        out_data   = data[indx:(indx+data_trs[i]),:]
        if verbose:
            print(f'[{ii}/{n_files_to_write}] writing modified data {out_path}')
        out_dir = osp.dirname(out_path)
        if not osp.exists(out_dir):
            print(f'[{ii}/{n_files_to_write}] --> Creating output folder first: {out_dir}')
            os.makedirs(out_dir)
        write_out(out_data,mask,header,file_format,out_path)        
        indx += data_trs[i]
