import os
import mne
import h5py
import json
import numpy as np
from collections import OrderedDict
from typing import Union, List
"""
 mne montage data class related
"""

def _validate_stimuli_dict(stimuli_dict:dict):
    for k in stimuli_dict:
        stim:dict = stimuli_dict[k]
        if not isinstance(stim, dict):
            raise ValueError(f'value for stim {k} should be a dict')
        for feat_k, feat_v in stim.items():
            if isinstance(feat_v, dict):
                assert all([s in feat_v for s in ['x', 'timeinfo', 'tag']])
            else:
                pass
                # pattern = r"_fs\d+$"
                # assert re.search(pattern, feat_k)
    return stimuli_dict  

def mne_montage_to_h5py_group(montage:mne.channels.DigMontage, f:h5py.File):
    montage_grp = f.require_group('montage')
    pos_dict = montage.get_positions()
    for k,v in pos_dict.items():
        # print(k)
        if k == 'ch_pos':
            chs, ch_coords = list(zip(
                *[
                    (ch, ch_coord) 
                for ch, ch_coord in v.items()
            ]))
            ch_coords = np.stack(ch_coords)
            chs_json_str = json.dumps(chs)
            # print(chs_json_str)
            t_ds = montage_grp.create_dataset(k, data = ch_coords)
            t_ds.attrs['chs_json_str'] = chs_json_str
        elif k == 'coord_frame':
            montage_grp.attrs['coord_frame'] = v
        else:
            if v is None:
                v = np.array([])
            montage_grp.create_dataset(k, data = v)
    return f
    
def mne_montage_from_h5py_group(f:h5py.File):
    pos_dict = {}
    montage_grp = f['montage']
    for k, v in montage_grp.items():
        if k == 'ch_pos':
            t_dict = OrderedDict()
            t_ds = montage_grp[k]
            ch_coords = t_ds[:]
            chs = json.loads(t_ds.attrs['chs_json_str'])
            for i_ch, ch in enumerate(chs):
                t_dict[ch] = ch_coords[i_ch]
            pos_dict[k] = t_dict
        else:
            # print(v.shape)
            if v.shape == (0,):
                pos_dict[k] = None
            else:
                pos_dict[k] = v[:]
    pos_dict['coord_frame'] = montage_grp.attrs['coord_frame']
    montage = mne.channels.make_dig_montage(**pos_dict)
    return montage


"""
 DataRecord class related
"""
def data_record_to_h5py_group(
    key: str,
    data: np.ndarray,
    stim_id: Union[str, int],
    meta_info:dict,
    srate: int,
    f:h5py.File
):
    root_grp = f.require_group(f'records/{key}')
    root_grp.create_dataset('data', data = data)
    root_grp.attrs['stim_id'] = stim_id
    root_grp.attrs['srate'] = srate

    meta_info_grp = root_grp.require_group('meta_info')
    for k,v in meta_info.items():
        if isinstance(v, np.ndarray):
            meta_info_grp.create_dataset(k, data=v)
        else:
            meta_info_grp.attrs[k] = v
    
    return f

def data_record_from_h5py_group(
    f:h5py.File
):
    data = f['data'][:]
    stim_id = f.attrs['stim_id']
    srate = int(f.attrs['srate'])

    meta_info_grp = f['meta_info']
    meta_info = {}
    for k,v in meta_info_grp.attrs.items():
        meta_info[k] = v
    
    for k,v in meta_info.items():
        meta_info[k] = v
    
    return dict(
        data = data, stim_id = stim_id, meta_info = meta_info, srate = srate
    )

"""
Stim Dict related
"""

def check_list_of_string(data:List[str]):
    assert isinstance(data, list)
    assert all([isinstance(i, str) for i in data])
    return data

def stim_dict_to_hdf5(
    filename:str,
    stim_dict: dict,
    attrs:dict = None,
):
    with h5py.File(filename, 'a') as hdf5f:
        _validate_stimuli_dict(stim_dict)
        for stim_id in stim_dict:
            grp = hdf5f.require_group(stim_id)
            for feat_name in stim_dict[stim_id]:
                assert feat_name not in grp
                data = stim_dict[stim_id][feat_name]
                if isinstance(data, np.ndarray):
                    dataset = grp.create_dataset(feat_name, data = data)
                elif isinstance(data, dict):
                    dataset = discrete_stim_to_hdf5(
                        feat_name=feat_name,
                        feat_dict=data,
                        hdf5f=grp
                    )
                else:
                    raise TypeError
                
                if attrs is not None:
                    dataset.attrs.update(attrs[stim_id][feat_name])

def stim_dict_from_hdf5(
    filename:str,
) -> dict:
    stim_dict = {}
    with h5py.File(filename, 'r') as hdf5f:
        for stim_id, stim_grp in hdf5f.items():
            stim_dict[stim_id] = {}
            for k, v in stim_grp.items():
                if isinstance(v, h5py.Dataset):
                    stim_dict[stim_id][k] = v[:]
                elif isinstance(v, h5py.Group):
                    stim_dict[stim_id][k] = discrete_stim_from_hdf5(v)
                else:
                    raise TypeError
    _validate_stimuli_dict(stim_dict)
    return stim_dict

def discrete_stim_to_hdf5(
    feat_name:str,
    feat_dict:dict,
    hdf5f:h5py.Group
) -> h5py.Group:
    """
    {
        'x': None,
        'tag':None,
        'timeinfo':None
    }
    """
    grp = hdf5f.require_group(feat_name)
    for k,v in feat_dict.items():
        if isinstance(v, np.ndarray):
            grp.create_dataset(k, data = v)
        elif check_list_of_string(v):
            string_list_to_hdf5(k, v, grp)
        else:
            raise TypeError
    return grp

def discrete_stim_from_hdf5(
    hdf5f:h5py.Group,
):
    stim_dict = {}
    for k,v in hdf5f.items():
        if isinstance(v, h5py.Dataset):
            v:h5py.Dataset
            if v.dtype == h5py.string_dtype(encoding='utf-8'):
                stim_dict[k] = string_list_from_hdf5(
                    v
                )
            elif v.dtype:
                stim_dict[k] = v[:]
        else:
            raise ValueError
    return stim_dict


def string_list_to_hdf5(
    dataset_name:str,
    strings: List[str],
    f:h5py.Dataset
):
    '''
    from chatGPT
    '''
    dt = h5py.string_dtype(encoding='utf-8')
    # Create dataset
    f.require_dataset(dataset_name, (len(strings),), dtype=dt, data = strings)
    return f

def string_list_from_hdf5(
    dt:h5py.Dataset
):
    return dt.asstr()[:].tolist()