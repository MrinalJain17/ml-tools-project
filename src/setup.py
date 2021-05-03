#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''A simple script to setup dataset files
'''
from pathlib import Path
import requests
import hashlib
import shutil

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

def download_data():
    '''Helper function to download files from Google Drive to Github/data/.
    - downloads datasets
    - checks md5
    - extracts dataset
    - preprocesses EURLex4k to proper format
    - moves and renames files
    - cleans up data directory
    
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
            shutil.rmtree(git_data+k)
            Path(dataset_zip).unlink()
        else:
            dataset = git_data + k + f'/{k}_data.txt'
            shutil.copy(dataset, git_data)
            shutil.rmtree(git_data+k)
            Path(dataset_zip).unlink()
        print(f'{k} dataset extracted & excess files removed')

def test_contextualbandits():
    try:
        import contextualbandits
    except:
        print("contextualbandits import failed")
        print("try: pip install contextualbandits")
    else:
        print("contextualbandits installed and functional")
            
def main():
    print(f'storing data to: {get_data_path()}')
    download_data()
    test_contextualbandits()
    
if __name__ == "__main__":
    main()
