import os
import numpy as np
import pickle
import pandas as pd

# Extract the list of file of dataset
feature_1s_dir = 'Extracted Feature/'
file_1s_dir = os.listdir(feature_1s_dir)
file_1s_dir.sort()

n_features = 62

## Data sort out
# Set zeros arrays for data
train_data = np.zeros((1, 62*5))
test_data = np.zeros((1, 62*5))
train_data_split = []
test_data_split = []
train_label = np.zeros((1, ))
test_label = np.zeros((1, ))

EEG_type = ['delta', 'theta', 'alpha', 'beta', 'gamma']


# Iterate the file to extract data into one full dataset
for item in file_1s_dir:
    npz_data = np.load(os.path.join(feature_1s_dir, item))
    
    # Whole train data
    train = pickle.loads(npz_data['train_data'])
    train_whole = train['delta']
    for key in EEG_type[1:]:
        train_whole = np.hstack((train_whole, train[key]))
    train_data = np.vstack((train_data, train_whole))

    # Whole test data
    test = pickle.loads(npz_data['test_data'])
    test_whole = test['delta']
    for key in EEG_type[1:]:
        test_whole = np.hstack((test_whole, test[key]))
    test_data = np.vstack((test_data, test_whole))

    # Train label
    label = npz_data['train_label']
    train_label = np.append(train_label, label)

    # Test label
    label = npz_data['test_label']
    test_label = np.append(test_label, label)

# Delete first zeros element we set at the beginning
train_data = train_data[1:]
test_data = test_data[1:]
train_label = train_label[1:]
test_label = test_label[1:]


# Iterate to extract 5 types of EEG signal DE
for key in EEG_type:
    train_split = np.zeros((1, 62))
    test_split = np.zeros((1, 62))
    
    for item in file_1s_dir:
        npz_data = np.load(os.path.join(feature_1s_dir, item))

        # Training dataset
        train = pickle.loads(npz_data['train_data'])
        train_split = np.vstack((train_split, train[key]))

        # Test dataset
        test = pickle.loads(npz_data['test_data'])
        test_split = np.vstack((test_split, test[key]))

    train_split = train_split[1:]
    test_split = test_split[1:]

    train_data_split.append(train_split)
    test_data_split.append(test_split)

train_data_split = np.array(train_data_split)
test_data_split = np.array(test_data_split)

# Save data
np.save('Available data/train_data.npy', train_data)
np.save('Available data/train_data_split.npy', train_data_split)
np.save('Available data/train_label.npy', train_label)
np.save('Available data/test_data.npy', test_data)
np.save('Available data/test_data_split.npy', test_data_split)
np.save('Available data/test_label.npy', test_label)