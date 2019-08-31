# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:31:29 2019

@author: Jin Dou
"""
from EEGDataSetClasses import bitalinoRawdata as glsEEG
from EEGDataSetClasses import AuditoryLabels, VisualLabels, AttentionLabels, BlinksCaliLabels
from preprocessing import CPreprocess as pre
import os
from DataIO import labelFileRecog, checkFolder

def labelsForEEGLab(dir_folder):
    dir_labels_mat = r"ForMatlab/"
    folder_path = dir_folder
    labelfiles = [file for file in os.listdir(folder_path) if file.endswith(".txt") and file.find('opensignal') == -1]

    for file in labelfiles:
        Type = labelFileRecog(folder_path+file)
        lbl = None
        if(Type == 'auditory'):
            lbl = AuditoryLabels()
        elif(Type == 'visual'):
            lbl = VisualLabels()
        elif(Type == 'attention'):
            lbl = AttentionLabels()
        elif(Type == 'blinkCali'):
            lbl = BlinksCaliLabels()
        else:
            raise Exception("doesn't recognize this type of label")
        #lbl.readFile(file,dir_dataset + dateForData + dir_stimuli)
        lbl.readFileForInfo(folder_path+file)
        checkFolder(folder_path + dir_labels_mat)
        lbl.writeInfoForMatlab(folder_path + dir_labels_mat)
