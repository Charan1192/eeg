import pandas as pd
import matplotlib.pyplot as plt
import mne
import numpy as np
import time
import pickle
import os.path

pd.options.display.float_format = '{:.4f}'.format

outPSD = []

# Define the subject numbers and stroke hemisphere information
allSubjectNumber = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
strokeHemisphere = [2, 2, 1, 2, 1, 2, 1, 2, 1,  1,  2,  1,  2,  1,  2,  1,  1,  1,  2,  2,  1]


for idx, isub in enumerate(allSubjectNumber):
    if isub in (12,17):
        continue
    substr = str(isub).zfill(2)
    print('______________')
    print('Subject #: ' + substr)
    print('______________')

    subPSDs = []

    for isesh in range(1, 4):
        print('Session:' + str(isesh))
        
        # Read EEG data for the current subject and session
        dataType = 'EEG'
        file_path = f'./data/ID_{substr}/{substr}_{dataType}_EVT_{isesh}_stroke_study_updated.csv'

        if not os.path.isfile(file_path):
            print(f"File {file_path} not found. Skipping...")
            continue
        df = pd.DataFrame()
        df = pd.read_csv(file_path, engine='python')
        df.reset_index(inplace=True)
        intervals = df['timestamps'].diff()
        if 2 <= isub <= 16:
            srate = round(256/intervals.mean())
        else:
            srate = round(1/intervals.mean())     #ask question
        if (isub == 12) & (isesh == 2): 
            df = df.astype({'timestamps': 'float'})

        if (isub == 15) & (isesh == 2): 
            df = df.astype({'timestamps': 'float'})

        if (isub == 16) & (isesh == 3): 
            df = df.astype({'timestamps': 'float'})

        if (isub == 17) & (isesh == 1): 
            df = df.astype({'timestamps': 'float', 'TP9': 'float', 'AF7': 'float', 'AF8': 'float', 'TP10': 'float', 'Right AUX': 'float'})
        
       #ch_names = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']  # EEG channel names
        #ch_types = ['eeg'] * 5  # EEG channel types
        

        #data = df[ch_names].values.T / 1000000  # Transpose and convert to seconds

        #rawData = mne.io.RawArray(data, info)

        #picks = mne.pick_types(rawData.info, eeg=True)

        #rawSpectra = rawData.compute_psd(picks=picks)
        #psds, freqs = rawSpectra.get_data(return_freqs=True, picks=picks)
        #ch_names = ['TP9', 'AF7', 'AF8', 'TP10', 'Right AUX']  # EEG channel names
        ch_names = ['TPBad', 'AFBad', 'AFGood', 'TPGood', 'Wrist']
        ch_types = ['eeg'] * 4 + ['ecg']
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=256)  #256,# Assuming sampling frequency is 1000 Hz
        info.set_montage('standard_1020',on_missing='ignore')
       
        print(df.head)
        # Drop unnecessary columns (excluding ACC and GYRO)
        if 12 <= isub <= 16:
             df = df.drop(['index', 'timestamps', 'Marker0'], axis=1)
        else:
             df = df.drop(['index', 'timestamps',  'Marker0'], axis=1)
        # Convert dataframe to numpy array and transpose
        data = df.to_numpy().transpose() / 1000000  # Assuming division by 1000000 is for conversion for volt i think

        # Switch data columns if strokeHemisphere is 1 (Right hemisphere)
        if strokeHemisphere[idx] == 1:
            print('Switching data columns for Right hemisphere infarcts')
            tempDataTP = data[0, :]
            tempDataAF = data[1, :]
            data[0, :] = data[3, :]
            data[1, :] = data[2, :]
            data[3, :] = tempDataTP
            data[2, :] = tempDataAF

        # Create RawArray object
        rawData = mne.io.RawArray(data, info)

        # Select EEG, ECG, and bio channels
        eeg_picks = mne.pick_types(rawData.info, eeg=True)

        # Compute power spectral density (PSD)
        rawSpectra = rawData.compute_psd(picks=eeg_picks)
        psds, freqs = rawSpectra.get_data(return_freqs=True, picks=eeg_picks)
        # Handle special cases
        if isub == 3 and isesh == 2:
            a = np.empty(np.shape(psds[-3:, :]))
            a.fill(np.finfo(float).eps)
            psds[-3:, :] = a

        if isub == 12 and isesh == 3:
            a = np.empty(np.shape(psds[5:8, :]))
            a.fill(np.finfo(float).eps)
            psds[5:8, :] = a

        if isub == 15 and isesh == 2:
            a = np.empty(np.shape(psds[5:, :]))
            a.fill(np.finfo(float).eps)
            psds[5:, :] = a

        if isub == 16 and isesh == 3:
            a = np.empty(np.shape(psds[-3:, :]))
            a.fill(np.finfo(float).eps)
            psds[-3:, :] = a

        if isub == 17 and isesh == 1:
            a = np.empty(np.shape(psds[:5, :]))
            a.fill(np.finfo(float).eps)
            psds[:5, :] = a
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=srate) 
        print(psds.shape)
        subPSDs.append(psds)

    outPSD.append(subPSDs)

# Convert to numpy array
npOut = np.array(outPSD)
print(npOut.size)
delta_band_indices = (0.5,3.9) # Define the indices for the Delta band
alpha_band_indices = (8,12) # Define the indices for the Alpha band
print(delta_band_indices)

# Find the integer indices of 'TPGood' and 'AFGood' in ch_names
TPGood_index = ch_names.index('TPGood')
AFGood_index = ch_names.index('AFGood')
TPBad_index = ch_names.index('TPBad')
AFBad_index = ch_names.index('AFBad')

dar_data = []
subject=[]
delta_indices = np.logical_and(freqs >= 0.5, freqs <= 4)
theta_indices = np.logical_and(freqs >= 4, freqs <= 7.9)
alpha_indices = np.logical_and(freqs >= 8, freqs <= 12.9)
beta_indices = np.logical_and(freqs >= 12, freqs <= 30.9)
print(delta_indices)
for sub in range(len(allSubjectNumber)):
    isub = allSubjectNumber[sub]
    print(isub)
    for h in range(3):
        selected_channels = npOut[sub, h, [TPGood_index, AFGood_index], :]

        # Step 2: Apply delta_indices to the selected channels
        delta_power_good1 = np.sum(selected_channels[:, delta_indices], axis=1)
        print(delta_power_good1)
        delta_power_good_tp = np.sum(npOut[sub, h, [TPGood_index], delta_indices])
        delta_power_bad_tp = np.sum(npOut[sub, h, [TPBad_index], delta_indices])
        delta_power_good_af = np.sum(npOut[sub, h, [AFGood_index], delta_indices])
        delta_power_bad_af = np.sum(npOut[sub, h, [AFBad_index], delta_indices])
        delta_power_good = np.sum(npOut[sub, h, [TPGood_index,AFGood_index], delta_indices])
        delta_power_bad = np.sum(npOut[sub, h, [TPBad_index,AFBad_index], delta_indices])
        # Calculate power in Alpha band for good and bad conditions
        alpha_power_good_tp = np.sum(npOut[sub, h, [TPGood_index], alpha_indices])
        alpha_power_bad_tp = np.sum(npOut[sub, h, [TPBad_index], alpha_indices])
        alpha_power_good_af = np.sum(npOut[sub, h, [AFGood_index], alpha_indices])
        alpha_power_bad_af = np.sum(npOut[sub, h, [AFBad_index], alpha_indices])
        alpha_power_good = np.sum(npOut[sub, h, [TPGood_index,AFGood_index], alpha_indices])
        alpha_power_bad = np.sum(npOut[sub, h, [TPBad_index,AFBad_index], alpha_indices])
        # Calculate power in beta band for good and bad conditions
        beta_power_good_tp = np.sum(npOut[sub, h, [TPGood_index], beta_indices])
        beta_power_bad_tp = np.sum(npOut[sub, h, [TPBad_index], beta_indices])
        beta_power_good_af = np.sum(npOut[sub, h, [AFGood_index], beta_indices])
        beta_power_bad_af = np.sum(npOut[sub, h, [AFBad_index], beta_indices])
        beta_power_good = np.sum(npOut[sub, h, [TPGood_index,AFGood_index], beta_indices])
        beta_power_bad = np.sum(npOut[sub, h, [TPBad_index,AFBad_index], beta_indices])
        # Calculate power in theta band for good and bad conditions
        theta_power_good_tp = np.sum(npOut[sub, h, [TPGood_index], theta_indices])
        theta_power_bad_tp = np.sum(npOut[sub, h, [TPBad_index], theta_indices])
        theta_power_good_af = np.sum(npOut[sub, h, [AFGood_index], theta_indices])
        theta_power_bad_af = np.sum(npOut[sub, h, [AFBad_index], theta_indices])
        theta_power_good = np.sum(npOut[sub, h, [TPGood_index,AFGood_index], theta_indices])
        theta_power_bad = np.sum(npOut[sub, h, [TPBad_index,AFBad_index], theta_indices])
        print("DAR good_af:",delta_power_good_af/alpha_power_good_af)
        print("DAR bad_af",delta_power_bad_af/alpha_power_bad_af)
        print("DAR log10good_af:",np.log10(delta_power_good_af/alpha_power_good_af))
        print("DAR log10bad_af:",np.log10(delta_power_bad_af/alpha_power_bad_af))
        print("DAR good_tp:",delta_power_good_tp/alpha_power_good_tp)
        print("DAR bad_tp",delta_power_bad_tp/alpha_power_bad_tp)
        print("DAR log10good_tp:",np.log10(delta_power_good_tp/alpha_power_good_tp))
        print("DAR log10bad_tp:",np.log10(delta_power_bad_tp/alpha_power_bad_tp))
        print("DAR  good:",delta_power_good/alpha_power_good)
        print("DAR  bad",delta_power_bad/alpha_power_bad)
        print("DAR log10good:",np.log10(delta_power_good/alpha_power_good))
        print("DAR log10bad:",np.log10(delta_power_bad/alpha_power_bad))
        dar_values = np.array([
                [delta_power_good_af / alpha_power_good_af, delta_power_bad_af / alpha_power_bad_af],
                [np.log10(delta_power_good_af / alpha_power_good_af), np.log10(delta_power_bad_af / alpha_power_bad_af)],
                [delta_power_good_tp / alpha_power_good_tp, delta_power_bad_tp / alpha_power_bad_tp],
                [np.log10(delta_power_good_tp / alpha_power_good_tp), np.log10(delta_power_bad_tp / alpha_power_bad_tp)],
                [delta_power_good / alpha_power_good, delta_power_bad / alpha_power_bad],
                [np.log10(delta_power_good / alpha_power_good), np.log10(delta_power_bad / alpha_power_bad)]
        ])
        values = np.array([
        [delta_power_good_tp, delta_power_bad_tp],
        [delta_power_good_af, delta_power_bad_af],
        [alpha_power_good_tp, alpha_power_bad_tp],
        [alpha_power_good_af, alpha_power_bad_af],
        [theta_power_good_tp, theta_power_bad_tp],
        [theta_power_good_af, theta_power_bad_af],
        [beta_power_good_tp, beta_power_bad_tp],
        [beta_power_good_af, beta_power_bad_af]
        ])
        subject.append([isub] + dar_values.flatten().tolist())
        dar_data.append([isub] + values.flatten().tolist())

# Create DataFrame for DAR values
df_dar = pd.DataFrame(subject, columns=['Subject', 'DAR_af_Good', 'DAR_af_Bad', 'DAR_log10_af_Good', 'DAR_log10_af_Bad', 'DAR_tp_Good', 'DAR_tp_Bad', 'DAR_log10_tp_Good', 'DAR_log10_tp_Bad', 'DAR_Good', 'DAR_Bad', 'DAR_log10_Good', 'DAR_log10_Bad'])

# Create DataFrame for power values
df_values = pd.DataFrame(dar_data, columns=['Subject', 'delta_tp_Good', 'delta_tp_Bad', 'delta_af_Good', 'delta_af_Bad', 'alpha_tp_Good', 'alpha_tp_Bad', 'alpha_af_Good', 'alpha_af_Bad', 'theta_tp_Good', 'theta_tp_Bad', 'theta_af_Good', 'theta_af_Bad', 'beta_tp_Good', 'beta_tp_Bad', 'beta_af_Good', 'beta_af_Bad'])
excel_path = r'E:\DAR_and_power_values11.xlsx'
with pd.ExcelWriter(excel_path) as writer:
    df_dar.to_excel(writer, sheet_name='DAR_values', index=False)
    df_values.to_excel(writer, sheet_name='Power_values', index=False)