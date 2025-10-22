import os
import mne
import h5py
import json
import numpy as np
from collections import OrderedDict
from StellarInfra import siIO
from tour.dataclass.io import (
    mne_montage_to_h5py_group,
    mne_montage_from_h5py_group,
    data_record_from_h5py_group,
    data_record_to_h5py_group,
    stim_dict_to_hdf5,
    stim_dict_from_hdf5
)
from tour.dataclass.dataset import Dataset, DataRecord
from StimRespFlow.DataStruct.DataSet import CDataSet, CDataRecord


current_folder = os.path.dirname(os.path.abspath(__file__))

def to_impulses(vectors, timestamps, f:float,padding_s = 0):
    '''
    # align the vectors into impulses with specific sampling rate 
    '''
    startTimes = timestamps[0]
    endTimes = timestamps[1]
    secLen = endTimes[-1] + padding_s
    nLen = np.ceil( secLen * f).astype(int)
    nDim = vectors.shape[0]
    out = np.zeros((nDim,nLen))
    
    timeIndices = np.round(startTimes * f).astype(int)
    out[:,timeIndices] = vectors
    return out

def test_save_mne_montage():
    output_fd = os.environ['box_root']
    montage = mne.channels.make_standard_montage('biosemi128')
    fig = montage.plot(show = False)
    fig.savefig(f"{current_folder}/target_montage.png")
    # pos_dict = montage.get_positions()
    with h5py.File(f"{output_fd}/Collab-Project/CompiledDataset/biosemi128_montage.h5", "w") as f:
        mne_montage_to_h5py_group(montage, f)
    
def test_load_montage_in_mne():
    output_fd = os.environ['box_root']
    with h5py.File(f"{output_fd}/Collab-Project/CompiledDataset/biosemi128_montage.h5", "r") as f:
        montage = mne_montage_from_h5py_group(f)
    fig = montage.plot(show = False)
    fig.savefig(f"{current_folder}/loaded_montage.png")


data_path = f"{os.environ['box_root']}/Collab-Project/CompiledDataset/ns.pkl"
dataset:CDataSet = CDataSet.load(siIO.loadObject(data_path))
# print(dataset.records[0])


dataset_name = 'lalorlab_oldman'
dataset_new = Dataset(
    name = dataset_name,
    srate = dataset.srate
)

stim_dict = dataset.stimuliDict
old_stim_ids = list(stim_dict.keys())
for old_stim_id in old_stim_ids:
    values = stim_dict[old_stim_id]
    # print(old_stim_id, values.keys())
    new_stim_id = old_stim_id.replace('phonemes','oldman')
    words = values['words']
    timeinfo = values['lex_sur']['timeinfo']
    x = values['lex_sur']['x']
    uniqueness_point = values['uni_pnt']['x']
    values['lexical_surprisal'] = {
        'tag': words,
        'timeinfo': timeinfo,
        'x': x
    }
    values['uniqueness_point'] = {
        'tag': words,
        'timeinfo': timeinfo,
        'x': uniqueness_point
    }
    values['lexical_surprisal_fs64'] = to_impulses(
        x,
        timeinfo,
        dataset.srate
    )
    ones = np.ones(x.shape)
    # print(x.shape, timeinfo.shape)
    values['word_onset_fs64'] = to_impulses(
        ones,
        timeinfo,
        dataset.srate
    )
    values['envelope_fs64'] = values['env']
    del values['lex_sur']
    del values['onset']
    del values['words']
    del values['env']
    del values['uni_pnt']
    stim_dict[new_stim_id] = stim_dict[old_stim_id]
    del stim_dict[old_stim_id]

# print(stim_dict)
for record in dataset.records:
    record:CDataRecord
    data = record.data
    old_info = record.descInfo
    stim_id = 'oldman' + str(old_info['stim'])
    trial_id = old_info['stim']
    subj_id = old_info['subj']
    meta_info = dict(
        subj_id = subj_id,
        trial_id = trial_id,
        dataset_name = dataset_name
    )
    record_new = DataRecord(
        data,
        stim_id,
        meta_info,
        srate = dataset.srate
    )
    dataset_new.append(record_new)


stim_dict_to_hdf5(
    f"{os.environ['box_root']}/Collab-Project/CompiledDataset/ns_unipnt_lexsur_env_onset.h5",
    stim_dict
)

stim_dict_new = stim_dict_from_hdf5(
    f"{os.environ['box_root']}/Collab-Project/CompiledDataset/ns_unipnt_lexsur_env_onset.h5",
)

# test_save_mne_montage()
# test_load_montage_in_mne()
    
dataset_new.dump(f"{os.environ['box_root']}/Collab-Project/CompiledDataset/ns.h5")
new_dataset_new = Dataset.load(f"{os.environ['box_root']}/Collab-Project/CompiledDataset/ns.h5")
