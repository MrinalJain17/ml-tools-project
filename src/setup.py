#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''A simple script to setup dataset files
'''
import re
import sys
import requests
import hashlib
import shutil
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

def parse_data(filename):
    '''Helper function to parse datasets; used to downsample MediamillReduced
    
    Returns
    -------
    features, labels : numpy.ndarrays
    '''
    with open(filename, "rb") as f:
        infoline = f.readline()
        infoline = re.sub(r"^b'", "", str(infoline))
        n_features = int(re.sub(r"^\d+\s(\d+)\s\d+.*$", r"\1", infoline))
        features, labels = load_svmlight_file(f, n_features=n_features, multilabel=True)
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)
    features = np.array(features.todense())
    features = np.ascontiguousarray(features)
    return features, labels

def get_data_path():
    '''Helper function to get/setup data path; should handle windows, mac, nix
    
    Returns
    -------
    path_to_data : string
    '''
    path = Path('.').resolve()
    path_string = path.absolute().as_posix()
    if 'src' in path_string:
        path = path.parent / 'data'
    elif 'data' in path_string:
        pass
    else:
        path = path / 'data'
    path_to_data = f'{path.absolute().as_posix()}/'
    Path(path_to_data).mkdir(parents=True, exist_ok=True)
    return path_to_data

def download_data(redownload=False):
    '''Helper function to download files from Google Drive to Github/data/.
    - downloads datasets
    - checks md5
    - extracts dataset
    - preprocesses EURLex4k & MediamillReduced to proper formats
    - moves and renames files
    - cleans up data directory
    
    redownload : boolean, default=False 
        redownloads and overwrites datasets; useful if they get corrupted
    
    Returns
    -------
    nothing... but you get shiny new dataset text files!
    '''
    # bypass google large file download warning
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    # save chunked files correctly
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    
    # verify downloaded file checksum
    def check_md5(file, checksum):
        m = hashlib.md5()
        with open(file, 'rb') as f:
            data = f.read()
            m.update(data)
        return m.hexdigest() == checksum
    
    url = "https://docs.google.com/uc?export=download"
    
    # Datset: (google_id, md5_checksum)
    ids = {'Mediamill': ('0B3lPMIHmG6vGY3B4TXRmZnZBTkk', '11cfdb9269ae5a301bb0c6f333cc6999'),
           'Bibtex'   : ('0B3lPMIHmG6vGcy1xM2pJZ09MMGM', '78db23bd26486d4188502abf6a39b856'),
           'Delicious': ('0B3lPMIHmG6vGdG1jZ19VS2NWRVU', 'e73730438e67ad179d724503e6770b1a'),
           'Eurlex'   : ('0B3lPMIHmG6vGU0VTR1pCejFpWjg', 'ec8feb2a9a0bd0f32d8f4c2b00725524')}

    git_data = get_data_path()
    session  = requests.Session()
    
    for k,v in ids.items():
        # do nothing if the dataset already exists and we don't want to redownload
        if Path(git_data+f'{k}_data.txt').exists() and (redownload==False):
            print(f'Skipping: {k}, dataset already exists')
            continue
        
        print(f'Downloading: {k}')
        response = session.get(url, params = {'id': v[0]}, stream=True)
        token = get_confirm_token(response)
    
        if token:
            params = {'id': v[0], 'confirm': token}
            response = session.get(url, params=params, stream=True)
    
        dataset_zip = git_data+k+'.zip'
        save_response_content(response, dataset_zip)
        assert check_md5(dataset_zip, v[1]), f'{k} download failed md5'
        
        # unzip
        shutil.unpack_archive(dataset_zip, git_data)
        
        # extract datasets & cleanup
        if k=='Eurlex':
            with open(git_data+f'{k}_data.txt', 'w') as outfile:
                # write manual header
                outfile.write(f'{15539+3809} 5000 3993\n')
                for fname in ['eurlex_train.txt', 'eurlex_test.txt']:
                    with open(git_data+f'{k}/'+fname) as infile:
                        # skip header
                        next(infile)
                        for line in infile:
                            outfile.write(line)
        else:
            dataset = git_data + k + f'/{k}_data.txt'
            shutil.copy(dataset, git_data)
        shutil.rmtree(git_data+k)
        Path(dataset_zip).unlink()
        print(f'{k} dataset extracted & excess files removed')
        
    # if MediamillReduced doesn't exist, or you want to 'redownload' it
    filename = git_data+f'MediamillReduced_data.txt'
    if (Path(filename).exists()==False) or redownload:
        print('Creating MediamillReduced')
        features, y = parse_data(git_data+'Mediamill_data.txt')
        # create MediamillReduced from Mediamill by removing 5 most common lables
        idx = np.argpartition(y.sum(axis=0), -5)[-5:]
        _y  = np.delete(y, idx, axis=1)
        
        # save in svmlight format
        dump_svmlight_file(features, _y, filename, multilabel=True)
        
        # prepend infoline to keep same format across datasets
        infoline = f'{features.shape[0]} {features.shape[1]} {_y.shape[1]}\n'
        with open(filename, 'r') as original: data = original.read()
        with open(filename, 'w') as modified: modified.write(infoline + data)
        
        # verify the file saved correctly
        features, y = parse_data("../data/MediamillReduced_data.txt")
        assert np.array_equal(_y, y), 'MediamillReduced_data.txt saved incorrectly'
    else:
        print('Skipping: MediamillReduced, dataset already exists')

def test_contextualbandits():
    try:
        import contextualbandits
    except:
        print("contextualbandits import failed")
        print("try: pip install contextualbandits")
    else:
        print("contextualbandits installed and functional")
            
def main(args=None):
    print(f'storing data to: {get_data_path()}')
    # sloppy argument handling; redownload=True if 'redownload' appears in sysargs
    download_data(redownload=np.any(['redownload' in x for x in args]))
    test_contextualbandits()
    
if __name__ == "__main__":
    args = sys.argv
    # sloppy argument handling for help: @TODO - correctly handle sysargs
    if np.any(['h' in x for x in args]):
        print("to redownload and overwrite your datasets, use:\n    python setup.py redownload")
    else:
        main(args)
