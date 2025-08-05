import os
import mne
import h5py
import json
import numpy as np
from collections import OrderedDict


def mne_montage_to_h5py_group(pos_dict:dict, f:h5py.File):
    montage_grp = f.require_group('montage')
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
