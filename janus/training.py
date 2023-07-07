import time
import csv
from tabulate import tabulate
import torch
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pandas.plotting import register_matplotlib_converters
from torch import nn
import copy

def main(settings_dict1, settings_dict2):

    PROJECT_TITLE = settings_dict1['project_title']
    PROJECT_PATH = settings_dict1['project_path']
    DEP_VAR = settings_dict1['cmip_dep_var']
    DEP_VAR_ORIG = settings_dict1['era_dep_var']
    INDEP_VARS = settings_dict1['training_indep_vars']
    TRAINING_DATA_PATH = '{0}{1}/{2}/'.format(PROJECT_PATH, PROJECT_TITLE, settings_dict1['era_export_path'])
    TRAINING_OUTPUTS_PATH = '{0}{1}/{2}/training/'.format(PROJECT_PATH, PROJECT_TITLE, settings_dict1['outputs_path'])
    MODELS_PATH = '{0}{1}/{2}/'.format(PROJECT_PATH, PROJECT_TITLE, settings_dict1['models_path'])
    LOGS_PATH = '{0}{1}/{2}/log/'.format(PROJECT_PATH, PROJECT_TITLE, settings_dict1['outputs_path'])
    TIME_TAG = settings_dict1['training_time_tag']
    OVERWRITE = settings_dict1['training_overwrite']
    OUTLINER_SD = settings_dict2['outliner_sd']
    TEST_SET_PROPORTION = settings_dict2['test_set_proportion']
    RANDOM_SEED = settings_dict2['random_seed']
    SEQ_LEN = settings_dict2['sequence_len']
    HIDDEN_DIM = settings_dict2['hidden_dim']
    NUM_LAYERS = settings_dict2['num_layers']
    OUTPUT_DIM = 1
    BATCH_SIZE = settings_dict2['batch_size']
    NUM_EPOCHS = settings_dict2['num_epochs']

    print('')
    print(('').center(80, '%'))
    print((' Training models ').center(80, '%'))
    print(('').center(80, '%'))
    print('')

    def create_folder(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    create_folder(TRAINING_OUTPUTS_PATH)
    create_folder(MODELS_PATH)
    create_folder(LOGS_PATH)

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

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
            # Initialize hidden state with zeros
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

            # Initialize cell state
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

            # One time step
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

            # Index hidden state of last time step
            out = self.fc(out[:, -1, :]) 
            return out

    # Generate log file
    with open('{0}train_logs.csv'.format(LOGS_PATH), 'a', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["INPUT_FILE", "UID", "RAND_SEED", "NUM_FEATURES", "HIDDEN_DIM", "NUM_LAYERS", "BATCH_SIZE", "NUM_EPOCHS", "SEQ_LEN", "OUTLINER_SD", "TRAIN_SIZE", "TEST_SIZE", "TRAIN_TIME", "TRAIN_MAE", "TRAIN_RMSE", "TEST_MAE", "TEST_RMSE"])

    filelist = os.listdir(TRAINING_DATA_PATH)
    filelist.sort()

    # print(filelist)
    i = 0

    for file_name in filelist:

        if TIME_TAG == True:
            modelid = '_' + str(int(time.time()))
        else:
            modelid = ''

        if OVERWRITE == False:
            if os.path.isfile('{0}{1}.pt'.format(MODELS_PATH, os.path.splitext(file_name)[0])):
                print(('MODEL ALREADY PROCESSED: {0} '.format(os.path.splitext(file_name)[0])))
                i = i + 1
                continue
        
        # Continue if the process has been stopped and there are existing model files
        i = i + 1

        print('')
        print((' MODEL ' + str(i) + '/' + str(len(filelist)) + ' ').center(60, '*'))
        print(file_name)
        
        # Remove extension from file name 
        file_name = os.path.splitext(file_name)[0]

        # Read data file
        raw_data = pd.read_csv(TRAINING_DATA_PATH + file_name + '.csv', parse_dates = ['time'],
                            index_col = 'time')
        df = raw_data.copy()
        
        if pd.isnull(df).all().any():
            print(('INPUT DATA CONTAINS AN ALL-NAN COLUMN: {0} '.format(os.path.splitext(file_name)[0])))
            continue

        # Replace missing values by interpolation
        def replace_missing (attribute):
            return attribute.interpolate(inplace=True)
        for (columnName, columnData) in df.items():
            replace_missing(df[columnName])

        # Remove outliner data
        up_b = df[DEP_VAR].mean() + OUTLINER_SD*df[DEP_VAR].std()
        low_b = df[DEP_VAR].mean() - OUTLINER_SD*df[DEP_VAR].std()
        df.loc[df[DEP_VAR] > up_b, DEP_VAR] = np.nan
        df.loc[df[DEP_VAR] < low_b, DEP_VAR] = np.nan
        df[DEP_VAR].interpolate(inplace=True)

        # Auto split data training and test sets
        train_size = int(len(df)*(1-TEST_SET_PROPORTION))
        train_dataset, test_dataset = df.iloc[:train_size],df.iloc[train_size:]

        # Print training and test sets dimensions
        print('')
        print(tabulate([[df.shape, train_dataset.shape, test_dataset.shape]], headers=['Pre-processed Data', 'Train Set', 'Test Set']))

        # Split training/test sets to X and y subsets
        X_train = train_dataset.loc[:,INDEP_VARS]
        y_train = train_dataset.loc[:,[DEP_VAR]]
        X_test = test_dataset.loc[:,INDEP_VARS]
        y_test = test_dataset.loc[:,[DEP_VAR]]

        INPUT_DIM = X_train.shape[1]

        # Define scalers
        scaler_x = MinMaxScaler(feature_range = (0,1))
        scaler_y = MinMaxScaler(feature_range = (0,1))
        # Fit scalerss
        input_scaler = scaler_x.fit(X_train)
        output_scaler = scaler_y.fit(y_train)
        # Apply scalers to training/test sets
        train_x_norm = input_scaler.transform(X_train)
        train_y_norm = output_scaler.transform(y_train)
        test_x_norm = input_scaler.transform(X_test)
        test_y_norm = output_scaler.transform(y_test)
        
        # Create sequence
        def create_dataset (X, y, time_steps):
            Xs, ys = [], []
            for i in range(len(X)-time_steps+1):
                v = X[i:i+time_steps, :]
                Xs.append(v)
                ys.append(y[i+time_steps-1])
            return np.array(Xs), np.array(ys)

        X_test, y_test = create_dataset(test_x_norm, test_y_norm,   
                                        SEQ_LEN)
        X_train, y_train = create_dataset(train_x_norm, train_y_norm, 
                                        SEQ_LEN)

        # Print training and test tensors
        print('')
        print(tabulate([[X_train.shape, y_train.shape, X_test.shape, y_test.shape]], headers=['X_train', 'y_train', 'X_test', 'y_test']))

        # Convert numpy data to torch tensors
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).float().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).float().to(device)

        # Convert torch tensors to datasets
        train = torch.utils.data.TensorDataset(X_train,y_train)
        test = torch.utils.data.TensorDataset(X_test,y_test)
        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=False)
           
        model = LSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM, num_layers=NUM_LAYERS).to(device)
        loss_fn = torch.nn.MSELoss(reduction='mean')
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train model
        print('')

        hist1 = np.zeros(NUM_EPOCHS)
        hist2 = np.zeros(NUM_EPOCHS)
        seq_dim = SEQ_LEN    # Number of steps to unroll

        start_time = time.time()

        n_epochs_stop = 10
        epochs_no_improve = 0
        early_stop = False
        min_val_loss = np.Inf

        for t in range(NUM_EPOCHS):
            val_loss = 0
            
            for xb,yb in train_loader:

                # Forward pass
                y_train_pred = model(xb)
                
                loss = loss_fn(y_train_pred, yb)
                hist1[t] = loss.item()

                # Zero out gradient to avoid accumulation between epochs
                optimiser.zero_grad()

                # Backward pass
                loss.backward()

                # Update parameters
                optimiser.step()

            for xb,yb in test_loader:

                # Forward pass
                y_test_pred = model(xb)

                loss = loss_fn(y_test_pred, yb)
  
                hist2[t] = loss.item()

                val_loss += loss

            val_loss = val_loss / len(test_loader)

            # Early stopping
            # Check if the validation loss is at a minimum
            if val_loss < min_val_loss:
                epochs_no_improve = 0
                min_val_loss = val_loss      
                best_model = copy.deepcopy(model.state_dict())

            else:
                epochs_no_improve += 1
            if t > 10 and epochs_no_improve == n_epochs_stop:
                # print('Epoch {0}: No improve over last {1} epochs. Early stopping!'.format(t, n_epochs_stop))
                print(("Epoch " + str(t+1) + " (ES)").ljust(15, ' ') , ("Train MSE: " + str(loss_fn(model(X_train), y_train).item())).rjust(20, ' ') , ("Test MSE: " + str(loss_fn(model(X_test), y_test).item())).rjust(20, ' '))
                early_stop = True
                model.load_state_dict(best_model)
                break

            if (t+1) % 10 == 0 and t !=0:
                print(("Epoch " + str(t+1)).ljust(15, ' ') , ("Train MSE: " + str(loss_fn(model(X_train), y_train).item())).rjust(20, ' ') , ("Test MSE: " + str(loss_fn(model(X_test), y_test).item())).rjust(20, ' '))

        train_time = time.time() - start_time

        # Make predictions
        y_train_pred = model(X_train).cpu()
        y_test_pred = model(X_test).cpu()
        y_train = y_train.cpu()
        y_test = y_test.cpu()

        # Invert predictions
        y_train_pred = scaler_y.inverse_transform(y_train_pred.detach().numpy())
        y_train = scaler_y.inverse_transform(y_train.detach().numpy())
        y_test_pred = scaler_y.inverse_transform(y_test_pred.detach().numpy())
        y_test = scaler_y.inverse_transform(y_test.detach().numpy())

        # calculate MAE and RMSE
        train_mae = mean_absolute_error(y_train[:,0], y_train_pred[:,0])
        train_rmse = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
        test_mae = mean_absolute_error(y_test[:,0], y_test_pred[:,0])
        test_rmse = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))

        print('')
        print(tabulate([[train_mae, train_rmse, test_mae, test_rmse]], headers=['Train MAE', 'Train RMSE', 'Test MAE', 'Test RMSE']))

        # Export predictions over both training and test sets for overfitting/performance analysis
        eval_dataset = pd.DataFrame(test_dataset[DEP_VAR].iloc[SEQ_LEN-1:])
        eval_dataset.columns = ['{0}_test_actual'.format(DEP_VAR)]
        eval_dataset['{0}_test_prediction'.format(DEP_VAR)] = pd.DataFrame(y_test_pred).values
        eval_dataset.to_csv('{0}{1}{2}.csv'.format(TRAINING_OUTPUTS_PATH, file_name, modelid))
        # fig = eval_dataset.plot(figsize=(10, 6)).get_figure()
        # fig.savefig('{0}{1}{2}_test_prediction.png'.format(TRAIN_OUTPUTS_PATH, file_name, modelid))
        # plt.close()
        
        # Save to log file
        with open('{0}train_logs.csv'.format(LOGS_PATH), 'a', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([file_name, modelid, RANDOM_SEED, INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, BATCH_SIZE, NUM_EPOCHS, SEQ_LEN, OUTLINER_SD, train_dataset.shape[0], test_dataset.shape[0], train_time, train_mae, train_rmse, test_mae, test_rmse ])

        # Save model
        torch.save(model.state_dict(), '{0}{1}{2}.pt'.format(MODELS_PATH, file_name, modelid))
        
        # Delete model to save memory
        del model