import nctoolkit as nc
import pandas as pd
import os
from datetime import datetime as dt
from tqdm import tqdm

import xarray as xa

def main(settings_dict):

    PROJECT_TITLE = settings_dict['project_title']
    PROJECT_PATH = settings_dict['project_path']
    CMIP_DATASET = settings_dict['cmip_dataset']
    CMIP_DATA_PATH = '{0}{1}/'.format(settings_dict['cmip_data_path'], CMIP_DATASET)
    ERA_DATA_PATH = settings_dict['era_data_path']
    CMIP_EXPORT_PATH = settings_dict['cmip_export_path']
    YEARS_RANGE = settings_dict['projection_years_range']
    INDEP_VARS = settings_dict['cmip_indep_vars']
    CROP_AREA = settings_dict['crop_area']
    SPLIT_SECTORS = settings_dict['split_sectors']

    print('')
    print(('').center(80, '%'))
    print((' Generating projection sets ').center(80, '%'))
    print(('').center(80, '%'))
    print('')

    nc.deep_clean()
    # nc.options(cores = 8)

    def create_folder(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    experiments = [ d for d in os.listdir(CMIP_DATA_PATH+'fetch/') if os.path.isdir(os.path.join(CMIP_DATA_PATH+'fetch/', d)) ]
    experiments.sort()
    
    for experiment in experiments:

        print('Processing experiment: {0}'.format(experiment))
        create_folder('{0}processed/{1}/{2}/'.format(CMIP_DATA_PATH, PROJECT_TITLE, experiment))
        create_folder('{0}{1}/{2}{3}/{4}/'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment))

        for ii in range(len(SPLIT_SECTORS)-1):
            print('Processing longitudinal sector: {0} to {1}'.format(SPLIT_SECTORS[ii], SPLIT_SECTORS[ii+1]))
            sector_lons = [SPLIT_SECTORS[ii], SPLIT_SECTORS[ii+1]]
            sector_lats = [CROP_AREA[2] , CROP_AREA[0]]

            print('Processing variables:')
            print(INDEP_VARS)

            for v in tqdm(INDEP_VARS):
                
                create_folder('{0}{1}/{2}{3}/{4}/{5}/'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment, v))

                print(sector_lons)
                nc.deep_clean()
                print(v)
                # print(os.listdir('{0}/{1}/{2}/'.format(CMIP_DATA_PATH+'fetch/', experiment, v)))
                ds = nc.open_data('{0}{1}/{2}/*.nc'.format(CMIP_DATA_PATH+'fetch/', experiment, v))
                # print(ds.contents)
                ds.merge(join='time')
                # ds.crop(lon=sector_lons, lat=sector_lats)
                ds.subset(variables = [v])
                # ds.run()
                ds.subset(years = range(YEARS_RANGE[0], YEARS_RANGE[1]))
                ds.run()
                
                print(ds.contents)
                # ds.drop(variable=['height'])    

                grid_ds_path = '{0}{1}/processed/'.format(ERA_DATA_PATH, PROJECT_TITLE)
                grid_ds_path =  grid_ds_path + os.listdir(grid_ds_path)[0]
                grid_ds = nc.open_data(grid_ds_path)
                # grid_ds.crop(lon=sector_lons, lat=sector_lats)
                grid_ds.run()
                ds.regrid(grid=grid_ds, method='nn')
                ds.run()


                ds.to_nc('{0}processed/{1}/{2}/{3}_{4}_{5}_{6}_nn.nc'.format(CMIP_DATA_PATH, PROJECT_TITLE, experiment, v, CMIP_DATASET, experiment, ii), overwrite = True)
                df = ds.to_dataframe()
                # print(df)

                # ds.to_nc('test_{}.nc'.format(v), overwrite=True)
                # print(df)
                # xa = df.to_xarray()
                # print(xa)
                # dss = nc.from_xarray(xa)
                # print(dss)
                # dss.to_nc('test2_{}.nc'.format(v), overwrite=True)
                
                if 'time_bnds' in df.columns:
                    df.drop(['time_bnds'], axis = 1, inplace=True)
                    df = df.loc[(df.index.get_level_values('bnds') == 1)]
                    df = df.droplevel(level=[1])

                # def check_unmasked(countries, pt_lat, pt_lon):
                #     pt = Point(pt_lon, pt_lat)
                #     for i in range(len(countries)):
                #         if pt.within(countries.values[i]):
                #             return True
                #     return False

                print('Processing the grid:')

                for idx, data in tqdm(df.groupby(['latitude', 'longitude'])):
                    latitude, longitude = idx

                    df_node = data.droplevel(['latitude', 'longitude'])
                    df_node = df_node.reset_index()
                    
                    def convert_to_dt(x):
                        return dt.strptime(str(x), '%Y-%m-%d %H:%M:%S')

                    df_node['time'] = df_node.time.apply(convert_to_dt)
                    df_node['time'] = pd.to_datetime(df_node['time']).dt.date
                    df_node.set_index(['time'], inplace=True)
                    # print(df_node)
                    df_node.to_csv('{0}{1}/{2}{3}/{4}/{5}/{6}_{7}.csv'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment, v, str(round(latitude, 1)), str(round(longitude, 1))))


    for experiment in experiments:

        print('{0}{1}/{2}{3}/{4}/'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment))
        create_folder('{0}{1}/{2}{3}/{4}/batch/'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment))

        nodes = []
        nodes_size = []
        for i , v in enumerate(INDEP_VARS):
            nodes.append([f for f in os.listdir('{0}{1}/{2}{3}/{4}/{5}/'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment, v)) if f.endswith('.csv')])
            nodes_size.append(len([f for f in os.listdir('{0}{1}/{2}{3}/{4}/{5}/'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment, v)) if f.endswith('.csv')]))  
            if len(set(nodes_size)) != 1:
                print('ERROR: Nodes are not equal between variables {0}'.format(experiment))
                nodes = []
                nodes_size = []
                break
        nodes = nodes[0]

        for node in tqdm(nodes):
            batch_df = pd.DataFrame()
            INDEP_VARS2 = INDEP_VARS[:]
            for v in INDEP_VARS:
                df_var = pd.read_csv('{0}{1}/{2}{3}/{4}/{5}/{6}'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment, v, node), parse_dates = ['time'],  infer_datetime_format=True, index_col = 'time')
                batch_df = pd.concat([batch_df, df_var], axis=1)

                # print(df_var)
                # print(batch_df)
                # batch_df = batch_df.join(df, how='outer')
                if batch_df.shape[0] != df_var.shape[0]:
                    print('ERROR: Incompatible time {0}'.format(node))
                    break
            # print(batch_df)

            if all(item in INDEP_VARS for item in ['uas', 'vas']):
                batch_df['ws'] = ((batch_df['uas'] ** 2) + (batch_df['vas'] ** 2)) ** 0.5
                INDEP_VARS2.remove('uas')
                INDEP_VARS2.remove('vas')
                INDEP_VARS2.append('ws')
            
            columns = INDEP_VARS2
            batch_df = batch_df[columns]

            batch_df.to_csv('{0}{1}/{2}{3}/{4}/batch/{5}'.format(PROJECT_PATH, PROJECT_TITLE, CMIP_EXPORT_PATH, CMIP_DATASET, experiment, node))