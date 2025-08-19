import os
import mne
import h5py
import json
import numpy as np
from collections import OrderedDict
from typing import Union
"""
 mne montage data class related
"""

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
 tray DataRecord class related
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