import json
import os


# Project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))


# Access and parse data
aligned_data_path = os.path.join(project_root, 'data', 'CRD3', 'aligned data')

# Number of training files wanted
train_ct = 1


# Fetch train_ct amount of training episodes
train_eps = []

with open(os.path.join(aligned_data_path, 'train_files'), 'r', encoding='utf-8') as f:
    train_eps_string = f.read()
    train_eps_string = train_eps_string.split()
    for train_ep in train_eps_string:
        train_eps.append(train_ep)
        train_ct-=1
        if train_ct == 0:
            break

# Fetch the file names corresponding to picked episodes

train_files = []

for train_ep in train_eps:
    train_ep_files = [f for f in os.listdir(os.path.join(aligned_data_path, 'c=4')) if f.startswith(train_ep) and os.path.isfile(os.path.join(os.path.join(aligned_data_path, 'c=4'), f))]
    train_files.extend(train_ep_files)


# TODO: Extract and process JSON