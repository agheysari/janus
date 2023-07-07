from tqdm import tqdm
from datetime import datetime as dt
import pandas as pd
import nctoolkit as nc
import os

def main(settings_dict):

    PROJECT_TITLE = settings_dict['project_title']
    PROJECT_PATH = settings_dict['project_path']
    ERA_DATA_PATH = settings_dict['era_data_path']
    ERA_EXPORT_PATH = settings_dict['era_export_path']
    YEARS_RANGE = settings_dict['training_years_range']
    INDEP_VARS = settings_dict['era_indep_vars']
    DEP_VAR = settings_dict['era_dep_var']
    RENAME_VARS = settings_dict['era_rename_vars']
    ERA_RESAMPLE = settings_dict['era_resample']

    print('')
    print(('').center(80, '%'))
    print((' Processing ERA5 data ').center(80, '%'))
    print(('').center(80, '%'))
    print('')

    def create_folder(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    create_folder('{0}{1}/{2}'.format(PROJECT_PATH, PROJECT_TITLE, ERA_EXPORT_PATH))
    create_folder('{0}{1}/processed/'.format(ERA_DATA_PATH, PROJECT_TITLE))

    nc.deep_clean()
    # nc.options(cores = 6)

    filelist_fetched = os.listdir(
        '{0}{1}/fetch/'.format(ERA_DATA_PATH, PROJECT_TITLE))
    filelist_fetched.sort()
    all_variables = INDEP_VARS + [DEP_VAR]

    for file in filelist_fetched:

        if os.path.isfile('{0}{1}/processed/{2}'.format(ERA_DATA_PATH, PROJECT_TITLE, file)):
            print(('FILE ALREADY PROCESSED: {0} '.format(file)))
            continue

        print(file)
        ds = nc.open_data(
            '{0}{1}/fetch/{2}'.format(ERA_DATA_PATH, PROJECT_TITLE, file))
        ds.set_precision("F32")
        # print(ds.contents)
        # print(ds.variables)

        if ERA_RESAMPLE > 1:
            print('Resampling')
            ds.resample_grid(ERA_RESAMPLE)
            ds.run()

        if all(item in INDEP_VARS for item in ['u10', 'v10']):
            ds.assign(ws=lambda x: ((x.u10 ** 2) + (x.v10 ** 2)) ** 0.5)
            ds.drop(variable=['u10', 'v10'])
            ds.run()

        if 'ssrd' in INDEP_VARS or 'strd' in INDEP_VARS:
            ds_mean = ds.copy()
            ds_max = ds.copy()

            ds_mean.drop(variable=['ssrd', 'strd'])
            ds_max.subset(variables=['ssrd', 'strd'])

            ds_mean.tmean(['year', 'month', 'day'])
            ds_max.tmax(['year', 'month', 'day'])

            ds_mean.run()
            ds_max.run()

            ds_max.assign(ssrd=lambda x: (x.ssrd / 86400))
            ds_max.assign(strd=lambda x: (x.strd / 86400))

            ds_batch = nc.merge(ds_mean, ds_max, match=[
                                'day', 'year', 'month'])
            ds_batch.run()
        else:
            ds_batch = ds.copy()
            ds_batch.merge(join='time')
            ds_batch.tmean(['year', 'month', 'day'])
            ds_batch.run()

        ds_batch.set_precision('F32')
        # print(ds_batch.contents)
        ds_batch.to_nc(
            '{0}{1}/processed/{2}'.format(ERA_DATA_PATH, PROJECT_TITLE, file))
        print(ds_batch.variables)

        nc.deep_clean()

    create_folder(
        '{0}{1}/{2}/'.format(PROJECT_PATH, PROJECT_TITLE, ERA_EXPORT_PATH))
    print('{0}{1}/processed/*.nc'.format(PROJECT_PATH, PROJECT_TITLE))
    nc.deep_clean()

    ds = nc.open_data(
        '{0}{1}/processed/*.nc'.format(ERA_DATA_PATH, PROJECT_TITLE))
    # print(ds.variables)
    # print(ds.contents)
    ds.set_precision("F32")

    if all(item in INDEP_VARS for item in ['u10', 'v10']):
        INDEP_VARS.remove('u10')
        INDEP_VARS.remove('v10')
        INDEP_VARS.append('ws')
    updated_vars = INDEP_VARS + [DEP_VAR]

    ds.subset(variables=updated_vars)
    ds.merge(join='time')
    ds.subset(years=range(YEARS_RANGE[0], YEARS_RANGE[1]))

    ds.run()

    df = ds.to_dataframe()
    print(df)

    df = df.loc[(df.index.get_level_values('bnds') == 1)]
    df = df.droplevel(level=[1])
    df = df[updated_vars]
    df.rename(RENAME_VARS, axis=1, inplace=True)

    if 'time_bnds' in df.columns:
        df.drop(columns=['time_bnds'], inplace=True)

    # print(df)
    # print(df_batch)

    print('Processing the grid:')

    for idx, data in tqdm(df.groupby(['latitude', 'longitude'])):
        latitude, longitude = idx
        df_node = data.droplevel(['latitude', 'longitude'])
        df_node = df_node.reset_index()
        df_node['time'] = pd.to_datetime(df_node['time']).dt.date
        df_node.set_index(['time'], inplace=True)
        df_node.to_csv('{0}{1}/{2}/{3}_{4}.csv'.format(PROJECT_PATH, PROJECT_TITLE, ERA_EXPORT_PATH, str(round(latitude, 1)), str(round(longitude, 1))))