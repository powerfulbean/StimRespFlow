# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:44:20 2025

@author: jdou3
"""
import re
import io
import copy
import h5py
import json
import itertools
# import h5py
from typing import List
import numpy as np

from .io import (
    data_record_from_h5py_group, data_record_to_h5py_group, _validate_stimuli_dict
)

#Note: please don't change the order, it matters for some functions using it
META_INFO_FORCED_FIELD = ['dataset_name', 'subj_id', 'trial_id'] 


def flatten_list_of_lists(list_of_lists:List[List]):
    return list(itertools.chain.from_iterable(list_of_lists))

def k_folds(n_trials, n_folds):
    id_trials = np.arange(n_trials)
    splits = np.array_split(id_trials, n_folds)
    for split_idx in range(len(splits)):
        # print('cv fold', split_idx)
        idx_val = splits[split_idx]
        idx_train = np.concatenate(splits[:split_idx] + splits[split_idx + 1 :])
        yield idx_train, idx_val


def _validate_meta_info(info:dict):
    assert all([k in info for k in META_INFO_FORCED_FIELD])
    for k, v in info.items():
        if isinstance(v, np.integer):
            info[k] = int(v)
    for k,v in info.items():
        assert isinstance(v, ((str, int, float, np.ndarray))), f"{k},{type(v)}"
    return info

def align_data(*arrs):
    arrs = list(arrs)
    #assume arr have shape [nChannel, nSamples]
    minLen  = min([arr.shape[1] for arr in arrs])
    for i, arr in enumerate(arrs):
        arrs[i] = arr[:, :minLen]
    return arrs
     
def i_split_kfold(tarList,cur_fold,n_folds, add_dev = True):
    ''' curFold starts from zero '''
    kfList = [i for i in k_folds(len(tarList), n_folds)]
    curTrainDevIdx = kfList[cur_fold][0]
    curTestIdx = kfList[cur_fold][1]
    curDevIdx = kfList[(cur_fold + 1) % n_folds][1]
    if add_dev:
        curTrainIdx = [i for i in curTrainDevIdx if i not in curDevIdx]
        curTrain = [tarList[i] for i in curTrainIdx]
        curDev = [tarList[i] for i in curDevIdx]
        curTest = [tarList[i] for i in curTestIdx]
        return curTrain, curDev, curTest 
    else:
        curTrainDev = [tarList[i] for i in curTrainDevIdx]
        curTest = [tarList[i] for i in curTestIdx]
        return curTrainDev, [], curTest

def k_fold(dataset:'Dataset', cur_fold, n_folds, split_by = 'trial_id', add_dev = True, if_shuffle = False, seed = 42):
    info_sets = sorted(
        list(set(
            [i.meta_info[split_by] for i in dataset.records]
    )))
    if if_shuffle:
        rng = np.random.default_rng(seed)
        inf_idxs = np.arange(len(info_sets))
        rng.shuffle(inf_idxs)
        info_sets = [info_sets[idx_] for idx_ in inf_idxs]
    info_train_list, info_dev_list, info_test_list = i_split_kfold(
        info_sets, cur_fold, n_folds, add_dev)
    print(info_train_list, info_dev_list, info_test_list)
    output = {}
    output['train'] = dataset.subset_by_info({split_by:info_train_list})
    if len(info_dev_list) > 0:
        output['dev'] = dataset.subset_by_info({split_by:info_dev_list})
    output['test'] = dataset.subset_by_info({split_by:info_test_list})
    return output


def dump_dict_contains_nparray(state_dict):
    output = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, value)
            t_value = buffer.getvalue()
        elif isinstance(value, dict):
            # print(key)
            t_value = dump_dict_contains_nparray(value)
        else:
            t_value = value
        output[key] = t_value
    return output

def load_dict_contains_nparray(state_dict):
    new_state = {}
    for k,v in state_dict.items():
        if isinstance(v, bytes):
            buffer = io.BytesIO(v)
            new_state[k] = np.load(buffer)
            # new_state[k] = np.frombuffer(v)
        elif isinstance(v, dict):
            # print(k)
            new_state[k] = load_dict_contains_nparray(v)
        else:
            new_state[k] = v
    return new_state

def encode_record_key(meta_info:dict):
    return "-".join(
        [str(meta_info[k]) for k in META_INFO_FORCED_FIELD]
    )

def decode_record_key(record_key:str):
    strs = record_key.split('-')
    return {
        k:v for k,v in zip(META_INFO_FORCED_FIELD, strs)
    }

class DataRecord:
    
    def __init__(self, data, stim_id, meta_info:dict, srate:int):
        self.srate = srate
        self.data = data
        self.stim_id = stim_id
        self.meta_info = _validate_meta_info(meta_info)

    def dump_to_dict(self):
        return dump_dict_contains_nparray(self.__dict__)
    
    def dump(self):
        record_key = encode_record_key(self.meta_info)
        return dict(
            key = record_key,
            data = self.data,
            stim_id = self.stim_id,
            meta_info = self.meta_info,
            srate = self.srate,
        )

    @classmethod
    def load(cls, new_state:dict):
        obj = cls(**new_state)
        return obj

    @classmethod
    def load_from_dict(cls, state:dict):
        new_state = load_dict_contains_nparray(state)
        obj = cls(**new_state)
        # for key in state:
            # obj.__dict__[key] = state[key]
        return obj
    
    def copy(self):
        new = DataRecord(
            self.data.copy(),
            self.stim_id,
            copy.deepcopy(self.meta_info),
            self.srate
        )
        return new 

class Dataset:
    
    # data and stim have the shape (nChannels, nSamples)
    # stim_id_cond: used when stimuli contains multiple conditions
    
    def __init__(self, name:str, srate:int):
        self.name = name
        self.srate = srate
        self.stim_feat_filter:list = []
        self.resp_chan_filter:list = []
        self.stim_id_cond:str|None = None
        self.meta_info_filter:dict = {}
        self._stimuli_dict:dict = {}
        self._records:List[DataRecord] = []
        self._preprocess_config = {}
    
    def copy(self):
        new_dataset = Dataset(
            self.name,
            self.srate
        )
        new_dataset.stim_feat_filter = copy.deepcopy(self.stim_feat_filter)
        new_dataset.resp_chan_filter = copy.deepcopy(self.resp_chan_filter)
        new_dataset.stim_id_cond = copy.deepcopy(self.stim_id_cond)
        new_dataset.meta_info_filter = copy.deepcopy(self.meta_info_filter)
        new_dataset._stimuli_dict = self._stimuli_dict
        new_dataset._records = [r_.copy() for r_ in self._records]
        return new_dataset

    @property    
    def stimuli_dict(self):
        return self._stimuli_dict
    
    @stimuli_dict.setter
    def stimuli_dict(self, x):
        self._stimuli_dict = _validate_stimuli_dict(x)
    
    @property
    def records(self) -> List[DataRecord]:
        if len(self.meta_info_filter) == 0:
            return self._records
        else:
            return self._filter_records_by_info(self._records, self.meta_info_filter)
    
    def _filter_records_by_info(self, records, meta_info_filter:dict):
        output = list()
        for record in records:
            if all(
                    [
                        record.meta_info[k] == v if np.isscalar(v) 
                            else record.meta_info[k] in v 
                                for k,v in meta_info_filter.items()
                    ]
            ):
                output.append(record)
        return output
    
    def append(self, record:DataRecord):
        assert record.srate == self.srate
        self._records.append(record)

    def _filter_stim_feat(self, stim_feat):
        new_stim_feat = {}
        if len(self.stim_feat_filter) == 0:
            stim_feat_filter = stim_feat.keys()
        else:
            stim_feat_filter = self.stim_feat_filter
        for i in stim_feat_filter:
            new_stim_feat[i] = stim_feat[i]
        return new_stim_feat
    
    def _filter_resp_chan(self, resp):
        if len(self.resp_chan_filter) > 0:
            idxArr = np.array(self.resp_chan_filter)
            output = resp[idxArr,:]
        else:
            output = resp
        return output

    def __getitem__(self, idx):
        record:DataRecord = self.records[idx]
        return self._unpack_record(record)
        
    def _unpack_record(self, record:DataRecord):
        stim_id, data = record.stim_id, record.data
        if isinstance(stim_id, dict):
            assert self.stim_id_cond is not None
            stim_id = stim_id[self.stim_id_cond]
        stim_feat = self._filter_stim_feat(self.stimuli_dict[stim_id])
        data = self._filter_resp_chan(data)
        return stim_feat, data, record.meta_info
    
    def __len__(self):
        return len(self.records)
    
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n  < len(self.records):
            self.n += 1
            return self.__getitem__(self.n-1)#self.records[self.n-1]
        else:
            raise StopIteration
    
    def to_pairs(self, ifT = True):
        allSubj = set([i.meta_info['subj_id'] for i in self.records])
        filterKey = lambda x: x.meta_info['subj_id']
        sortKey = lambda x : (
            x.meta_info['dataset_name'],
            x.meta_info['subj_id'], 
            x.meta_info['trial_id'], 
        )
        
        records = sorted(self.records, key = sortKey)
        
        transpose = lambda *arrs: [arr.T for arr in arrs]
        def catstimarr(stim:dict):
            keys = stim.keys()
            # print(keys)
            assert all([stim[k].shape[0] < stim[k].shape[1] for k in keys if isinstance(stim[k], np.ndarray)])
            stim = [stim[k] for k in keys if isinstance(stim[k], np.ndarray)]
            stim = align_data(*stim)
            stim = np.concatenate(stim, axis = 0)
            # print(stim.shape)
            return stim

        stims_subj = []
        resps_subj = []
        infoss = []
        ks = []
        for k, grp in itertools.groupby(records, filterKey):
            stims, resps, infos = list(zip(*[self._unpack_record(g) for g in grp]))
            # print(infos)
            stims = list(map(catstimarr, stims))
            stims, resps = list(zip(*map(align_data, stims, resps)))
            if ifT:
                stims, resps = list(zip(*map(transpose, stims, resps)))
            stims_subj.append(stims)
            resps_subj.append(resps)
            infoss.append(infos)
            ks.append(k)

        return stims_subj, resps_subj, ks, infoss
    
    def to_pairs_iter(self,sortKey = None):
        allSubj = set([i.meta_info['subj_id'] for i in self.records])
        
        filterKey = lambda x: x.meta_info['subj_id']
        if sortKey is None:
            sortKey = lambda x : (
                x.meta_info['dataset_name'],
                x.meta_info['subj_id'], 
                x.meta_info['trial_id'], 
            )
        
        records = sorted(self.records, key = sortKey)

        for k, grp in itertools.groupby(records, filterKey):
            stims, resps, infos = list(zip(*[self._unpack_record(g) for g in grp]))
            yield stims, resps, infos, k



    def k_fold(self, cur_fold, n_folds, split_by, add_dev = True, if_shuffle = False):
        return k_fold(self, cur_fold, n_folds, split_by, add_dev=add_dev, if_shuffle = if_shuffle)
    
    def subset_by_info(self,meta_info_filter):
        records = self._filter_records_by_info(
            self._records, meta_info_filter)
        state_dict = self.dump()
        state_dict['_records'] = [l.dump() for l in records]
        return self.__class__.load(state_dict)

    def dump_record(self, file_path, record:DataRecord):
        with h5py.File(file_path, "a") as f:
            data_record_to_h5py_group(
                f = f,
                **record.dump(),
            )

    def dump_attr(self, file_path):
        with h5py.File(file_path, "a") as f:
            f.attrs["name"] = self.name
            f.attrs["srate"] = self.srate
            preprocess_config = json.dumps(self._preprocess_config)
            f.attrs["preprocess_config_str"] = preprocess_config

    def dump(self, file_path):
        with h5py.File(file_path, "w") as f:
            f.attrs["name"] = self.name
            f.attrs["srate"] = self.srate
            preprocess_config = json.dumps(self._preprocess_config)
            f.attrs["preprocess_config_str"] = preprocess_config
            for record in self._records:
                data_record_to_h5py_group(
                    f = f,
                    **record.dump(),
                )
    
    @classmethod
    def load(cls, file_path):
        new_dataset = None
        with h5py.File(file_path, "r") as f:
            new_dataset = cls(
                name = str(f.attrs['name']),
                srate = int(f.attrs['srate']),
            )
            for k, grp in f['records'].items():
                record_dict = data_record_from_h5py_group(grp)
                new_record = DataRecord(**record_dict)
                new_dataset.append(new_record)
        return new_dataset

    def dump_to_dict(self):
        output = {}
        output['_records'] = [l.dump() for l in self._records]
        for k,v in self.__dict__.items():
            if k != '_records':
                if isinstance(v, dict):
                    output[k] = dump_dict_contains_nparray(v)
                else:
                    output[k] = v
        return output

    @classmethod
    def load_from_dict(cls, state):
        output = cls(name = state['name'], srate = state['srate'])
        for k,v in state.items():
            if k == '_records':
                output.__dict__['_records'] = [DataRecord.load(l) for l in state[k]]
            else:
                if isinstance(v, dict):
                    output.__dict__[k] = load_dict_contains_nparray(v)
                else:
                    output.__dict__[k] = state[k]
        return output
    
    @classmethod
    def load_subject(cls, file_path, subject_id):
        with h5py.File(file_path, "r") as f:
            all_keys = list(f['records'].keys())
            all_keys = sorted(all_keys, key = lambda x: [decode_record_key(x)[k] for k in META_INFO_FORCED_FIELD])
            cnter = 1
            new_dataset = cls(
                name = str(f.attrs['name']),
                srate = int(f.attrs['srate']),
            )
            for key_idx, key in enumerate(all_keys):
                if decode_record_key(key)['subj_id'] == subject_id:
                    record_dict = data_record_from_h5py_group(f['records'][key])
                    new_record = DataRecord(**record_dict)
                    new_dataset.append(new_record)
            return new_dataset
        
    @classmethod
    def iter_load(cls, file_path, n_subjs = 10):
        with h5py.File(file_path, "r") as f:
            all_keys = list(f['records'].keys())
            all_keys = sorted(all_keys, key = lambda x: [decode_record_key(x)[k] for k in META_INFO_FORCED_FIELD])
            # print(all_keys)
            cnter = 1
            new_dataset = cls(
                name = str(f.attrs['name']),
                srate = int(f.attrs['srate']),
            )
            for key_idx, key in enumerate(all_keys):

                record_dict = data_record_from_h5py_group(f['records'][key])
                new_record = DataRecord(**record_dict)
                new_dataset.append(new_record)


                if key_idx == len(all_keys)-1:
                    yield new_dataset
                else:
                    current_subj_id = decode_record_key(key)['subj_id']
                    next_subj_id = decode_record_key(all_keys[key_idx+1])['subj_id']

                    if current_subj_id != next_subj_id:
                        if cnter >= n_subjs:
                            yield new_dataset
                            new_dataset = cls(
                                name = str(f.attrs['name']),
                                srate = int(f.attrs['srate']),
                            )
                            cnter = 1
                        else:
                            cnter += 1