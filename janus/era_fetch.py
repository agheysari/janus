import os
import cdsapi

def main(settings_dict):

    ANALYSIS_TITLE = settings_dict['era_analysis']
    PROJECT_TITLE = settings_dict['project_title']
    ERA_DATA_PATH = settings_dict['era_data_path']
    YEARS_RANGE = settings_dict['training_years_range']
    AREA = settings_dict['crop_area']
    VARIABLES = settings_dict['era_download_vars']

    query_dict = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': None,
        'year': None,
        'month': None,
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': None,
    }

    years_range_backward = range(YEARS_RANGE[1]-1, YEARS_RANGE[0]-1, -1)
    months_range_backward = [
        '12', '11', '10',
        '09', '08', '07',
        '06', '05', '04',
        '03', '02', '01',
    ]

    print('')
    print(('').center(80, '%'))
    print((' Fetching ERA5 data from Copernicus ').center(80, '%'))
    print(('').center(80, '%'))
    print('')

    def create_folder(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    create_folder("{0}{1}/fetch/".format(ERA_DATA_PATH, PROJECT_TITLE))

    for year in years_range_backward:
        for month in months_range_backward:
            print(year)
            query_dict['variable'] = VARIABLES
            query_dict['year'] = [year]
            query_dict['month'] = [month]
            query_dict['area'] = AREA
            file_name = '{0}{1}/fetch/{1}_hr_{2}_{3}.nc'.format(
                ERA_DATA_PATH, PROJECT_TITLE, str(year), str(month))
            if os.path.isfile(file_name):
                print(
                    'Processing {0}/{1} - Already exists!'.format(year, month))
            else:
                print('Processing {0}/{1}'.format(year, month))
                c = cdsapi.Client()
                c.retrieve(ANALYSIS_TITLE, query_dict, file_name)
