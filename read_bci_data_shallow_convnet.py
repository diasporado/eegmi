from mne import Epochs, pick_types, find_events
from mne.decoding import Scaler
from mne.io import concatenate_raws, read_raw_edf
import numpy as np
import pandas as pd
import os.path as op
import glob
import scipy.io as sio
import pickle as pk

def load_raw(training=False):

    raw_edf = []
    
    if training:
        path = op.join('data_i2r', 'BCI_IV_2a', 'TrainingSet')
    if not training:
        path = op.join('data_i2r', 'BCI_IV_2a', 'TestingSet')
        label_path = op.join('data_i2r', 'BCI_IV_2a', 'true_labels')
        label_files_list = glob.glob(label_path + '/*E.mat')
        label_subj = [ int(f.split('A0')[1][0]) for f in label_files_list ]
        
    file_list = glob.glob(path + '/*.gdf')
    subjects = [ int(f.split('A0')[1][0]) for f in file_list ]
    
    if not training:
        label_subj = [ np.argwhere(np.array(label_subj)==subjects[i])[0][0]
                    for i in range(len(subjects))]
    
    raw_files = [read_raw_edf(raw_fnames, eog=['EOG-left','EOG-central','EOG-right'],
                              preload=True, stim_channel='auto')for raw_fnames in file_list]
    raw_edf.extend(raw_files)

    return raw_edf, subjects


def raw_to_data(raw_edf, training=False, drop_rejects=True, subj=None):
    
    tmin, tmax = -0.5, 4.
    X, y = [], []

    stim_code = dict([(32766,1),(769,2), (770,3), (771,4), (772,5),(783,6),(276,7),(277,8),(768,9),
                      (1023,10),(1072,11)])
    
    if training:
        path = op.join('data_i2r', 'BCI_IV_2a', 'TrainingSet')
    if not training:
        path = op.join('data_i2r', 'BCI_IV_2a', 'TestingSet')
        label_path = op.join('data_i2r', 'BCI_IV_2a', 'true_labels')
        label_files_list = glob.glob(label_path + '/*E.mat')
        label_subj = [ int(f.split('A0')[1][0]) for f in label_files_list ]
        
    file_list = glob.glob(path + '/*.gdf')
    subjects = [ int(f.split('A0')[1][0]) for f in file_list ]
    
    if not training:
        label_subj = [ np.argwhere(np.array(label_subj)==subjects[i])[0][0]
                    for i in range(len(subjects))]
    
    event_id = dict()
    events_from_edf = []
    sampling_frequency = raw_edf._raw_extras[0]['max_samp']
    original_event = raw_edf.find_edf_events()
    annot_list = list(zip(original_event[1], original_event[4], original_event[2]))
    
    # Remove rejected trials from events
    if drop_rejects:
        annot_list = pd.DataFrame(annot_list)
        rejected = annot_list[0].isin(annot_list[annot_list[2] == 1023][0])
        accepted_trials_index = [True] * 288
        ind=-1
        for row in annot_list.itertuples():
            if row[3] == 1023:
                rejected.loc[row[0]+1] = True
                accepted_trials_index[ind] = False
            if row[3] == 768:
                ind = ind + 1
            
    annot_list = annot_list[~rejected]
    annot_list = list(zip(annot_list[0], annot_list[1], annot_list[2]))
    
    events_from_edf.extend(annot_list)
    events_from_edf = np.array(events_from_edf)

    events_arr = np.zeros(events_from_edf.shape, dtype=int)
    for (i, i_event) in enumerate(events_from_edf):

        index = int((float(i_event[0])) * sampling_frequency)

        events_arr[i,:] = index,0,stim_code[int(i_event[2])]
        i=i+1

    # strip channel names of "." characters
    raw_edf.rename_channels(lambda x: x.strip('.'))
    #create Event dictionary based on File
    events_in_edf = [event[2] for event in events_arr[:]]
    if(events_in_edf.__contains__(2)):
        event_id['LEFT_HAND'] = 2
    if (events_in_edf.__contains__(3)):
        event_id['RIGHT_HAND'] = 3
    if (events_in_edf.__contains__(4)):
        event_id['FEET'] = 4
    if (events_in_edf.__contains__(5)):
        event_id['TONGUE'] = 5
    if (events_in_edf.__contains__(6)):
        event_id['CUE_UNKNOWN'] = 6

    # Apply band-pass filter
    raw_edf.filter(4., 40., fir_design='firwin', skip_by_annotation='edge')   # 4-40Hz

    picks = pick_types(raw_edf.info, meg=False, eeg=True, 
                       stim=False, eog=False, exclude='bads')

    # Read epochs (train will be done only between -0.5 and 4s)
    # Testing will be done with a running classifier

    epochs = Epochs(raw_edf, events_arr, event_id, tmin, tmax, proj=True, picks=picks,
            baseline=None, preload=True)
    
    X = epochs.get_data().transpose(0,2,1)
    X_shape = X.shape

    if training:
        oScaler = Scaler(scalings='mean').fit(X.flatten().reshape(-1,1))
        pk.dump(oScaler,open("./shallow_convnet/subject{}_oscaler.pk".format(subjects[subj]),'wb'))
    else:
        oScaler = pk.load(open("./shallow_convnet/subject{}_oscaler.pk".format(subjects[subj]),'rb'))
        
    y = epochs.events[:, 2] - 2
    # X = oScaler.transform(X.flatten().reshape(-1,1))
    X = X.reshape(X_shape)

    if training:
        return X, y
    else:
        y = sio.loadmat(label_files_list[label_subj[subj]])['classlabel'].flatten()
        y = np.array([ i - 1 for i in y ])
        if drop_rejects:
            y_drop = [ i for i in range(288) if not accepted_trials_index[i] ]
            y = np.delete(y, y_drop, None)
        return X, y
        
        