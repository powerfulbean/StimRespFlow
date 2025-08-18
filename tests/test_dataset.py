import os
import mne
import h5py
import json
import numpy as np
from collections import OrderedDict
from StellarInfra import siIO
from tray.dataclass.io import (
    mne_montage_to_h5py_group,
    mne_montage_from_h5py_group
)

current_folder = os.path.dirname(os.path.abspath(__file__))

def test_save_mne_montage():
    output_fd = os.environ['box_root']
    montage = mne.channels.make_standard_montage('biosemi128')
    fig = montage.plot(show = False)
    fig.savefig(f"{current_folder}/target_montage.png")
    pos_dict = montage.get_positions()
    with h5py.File(f"{output_fd}/Collab-Project/CompiledDataset/biosemi128_montage.h5", "w") as f:
        mne_montage_to_h5py_group(pos_dict, f)
    
def test_load_montage_in_mne():
    output_fd = os.environ['box_root']
    with h5py.File(f"{output_fd}/Collab-Project/CompiledDataset/biosemi128_montage.h5", "r") as f:
        montage = mne_montage_from_h5py_group(f)
    fig = montage.plot(show = False)
    fig.savefig(f"{current_folder}/loaded_montage.png")


data_path = f"{os.environ['box_root']}/Collab-Project/CompiledDataset/ns.pkl"
dataset = siIO.loadObject(data_path)
print(dataset)
# test_save_mne_montage()
# test_load_montage_in_mne()