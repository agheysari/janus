import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch import nn

def main(settings_dict1, settings_dict2):

    PROJECT_TITLE = settings_dict1['project_title']
    PROJECT_PATH = settings_dict1['project_path']
    CMIP_DATASET = settings_dict1['cmip_dataset']
    DEP_VAR = settings_dict1['cmip_dep_var']
    DEP_VAR_ORIG = settings_dict1['era_dep_var']
    INDEP_VARS = settings_dict1['training_indep_vars']
    TRAINING_DATA_PATH = '{0}{1}/{2}/'.format(PROJECT_PATH, PROJECT_TITLE, settings_dict1['era_export_path'])
    MODELS_PATH = '{0}{1}/{2}/'.format(PROJECT_PATH, PROJECT_TITLE, settings_dict1['models_path'])
    PROJECTION_INPUTS_PATH = '{0}{1}/{2}{3}/'.format(PROJECT_PATH, PROJECT_TITLE, settings_dict1['cmip_export_path'], CMIP_DATASET)
    PROJECTION_OUTPUTS_PATH = '{0}{1}/{2}projection/{3}/'.format(PROJECT_PATH, PROJECT_TITLE, settings_dict1['outputs_path'], CMIP_DATASET)
    OVERWRITE = settings_dict1['projection_overwrite']
    OUTLINER_SD = settings_dict2['outliner_sd']
    TEST_SET_PROPORTION = settings_dict2['test_set_proportion']
    RANDOM_SEED = settings_dict2['random_seed']
    SEQ_LEN = settings_dict2['sequence_len']
    HIDDEN_DIM = settings_dict2['hidden_dim']
    NUM_LAYERS = settings_dict2['num_layers']
    OUTPUT_DIM = 1

    print('')
    print(('').center(80, '%'))
    print((' Projection forward pass ').center(80, '%'))
    print(('').center(80, '%'))
    print('')
  
    # Manual random seed for reproducibility
    # np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Define model class
    class LSTM(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :]) 
            return out

    def get_xy_train(TRAIN_DATA_PATH, file_name, Y_COL, OUTLINER_SD, TEST_SET_PROPORTION):

        # Read data file
        train_dataset = pd.read_csv(TRAIN_DATA_PATH + file_name + '.csv', parse_dates = ['time'],
                            index_col = 'time')

        # Replace missing values by interpolation
        def replace_missing (attribute):
            return attribute.interpolate(inplace=True)
        for (columnName, columnData) in train_dataset.items():
            replace_missing(train_dataset[columnName])

        # Remove outliner data
        up_b = train_dataset[Y_COL].mean() + OUTLINER_SD*train_dataset[Y_COL].std()
        low_b = train_dataset[Y_COL].mean() - OUTLINER_SD*train_dataset[Y_COL].std()
        train_dataset.loc[train_dataset[Y_COL] > up_b, Y_COL] = np.nan
        train_dataset.loc[train_dataset[Y_COL] < low_b, Y_COL] = np.nan
        train_dataset[Y_COL].interpolate(inplace=True)

        # Split data training and test sets
        train_size = int(len(train_dataset)*(1-TEST_SET_PROPORTION))
        train_dataset = train_dataset.iloc[:train_size]

        # Split training/test sets to X and y subsets
        X_train = train_dataset.loc[:,INDEP_VARS]
        y_train = train_dataset.loc[:,[DEP_VAR]]

        return X_train, y_train

    def create_folder(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    experiments = [ d for d in os.listdir(PROJECTION_INPUTS_PATH) if os.path.isdir(os.path.join(PROJECTION_INPUTS_PATH, d)) ]

    print(experiments)
    
    for experiment in experiments:

        print('Processing experiment: {0}'.format(experiment))
        create_folder('{0}{1}/'.format(PROJECTION_OUTPUTS_PATH, experiment))   

        filelist = os.listdir(MODELS_PATH)
        filelist.sort()

        i = 0
        print('Processing nodes:')

        for file_name in filelist:
            
            i = i + 1
            print("{0}/{1}".format(i, len(filelist)))

            if OVERWRITE == False:
                if os.path.isfile('{0}{1}/{2}.csv'.format(PROJECTION_OUTPUTS_PATH, experiment, os.path.splitext(file_name)[0])):
                    print(('MODEL ALREADY PROCESSED: {0} {1}'.format(os.path.splitext(file_name)[0], experiment)))
                    continue

            # Remove extension from file name 
            file_name = os.path.splitext(file_name)[0]
            
            # Read projection data file
            projection_dataset = pd.read_csv('{0}{1}/batch/{2}.csv'.format(PROJECTION_INPUTS_PATH, experiment, file_name), parse_dates = ['time'],
                                index_col = 'time')

            # Replace missing values by interpolation
            def replace_missing (attribute):
                return attribute.interpolate(inplace=True)
            for (columnName, columnData) in projection_dataset.items():
                replace_missing(projection_dataset[columnName])

            X_train, y_train = get_xy_train(TRAINING_DATA_PATH, file_name, DEP_VAR, OUTLINER_SD, TEST_SET_PROPORTION)
            X_projection = projection_dataset
            
            INPUT_DIM = X_train.shape[1]        

            # Define scalers
            scaler_x = MinMaxScaler(feature_range = (0,1))
            scaler_y = MinMaxScaler(feature_range = (0,1))

            # Fit scalerss
            input_scaler = scaler_x.fit(X_train)
            output_scaler = scaler_y.fit(y_train)

            # Apply scalers to project set
            projection_x_norm = input_scaler.transform(X_projection)
                
            # Create sequence
            def create_dataset (X, time_steps = 1):
                Xs = []
                for i in range(len(X)-time_steps):
                    v = X[i:i+time_steps, :]
                    Xs.append(v)
                return np.array(Xs)

            X_projection = create_dataset(projection_x_norm, SEQ_LEN)

            # Convert numpy data to torch tensors
            X_projection = torch.from_numpy(X_projection).float()
        
            model = LSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS)
            model.load_state_dict(torch.load('{0}{1}.pt'.format(MODELS_PATH, file_name)))
            model.eval()
            
            # Make predictions
            y_projection_pred = model(X_projection)

            # Invert predictions
            y_projection_pred = scaler_y.inverse_transform(y_projection_pred.detach().numpy())
        
            # Plot evaluation (time series)
            eval_dataset = pd.DataFrame(projection_dataset.iloc[SEQ_LEN:])
            eval_dataset[DEP_VAR] = pd.DataFrame(y_projection_pred).values
            eval_dataset.to_csv('{0}{1}/{2}.csv'.format(PROJECTION_OUTPUTS_PATH, experiment, file_name))

            # fig = eval_dataset.plot(figsize=(10, 6)).get_figure()
            # fig.savefig('{0}{1}/{2}_projection.png'.format(PROJECTION_OUTPUTS_PATH, experiment, file_name))
            
            plt.close()
            
            del model



