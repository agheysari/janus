# Uncomment to import each part. Uncommenting all at the same time is not suggested due to interference between libraries.

# import era_fetch
# import era_preprocess
# import training
# import cmip_canrcm_preprocess
# import projection

project_settings = {
    "project_title": "nunavik_era5land",
    "era_analysis": "reanalysis-era5-land",
    "cmip_dataset": "canrcm4_nam",
    "crop_area": [63, -79, 55, -57],
    "split_sectors": [-80, -65, -55],
    "training_years_range": [1990, 1991],
    "projection_years_range": [2007, 2009],
    'era_resample': 2,
    "era_download_vars": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "skin_temperature",
        "soil_temperature_level_1",
        "snow_depth",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
    ],
    "era_indep_vars": ["t2m", "sde", "u10", "v10", "ssrd", "strd"],
    "era_dep_var": "stl1",
    "era_rename_vars": {
        "t2m": "tas",
        "sde": "snd",
        "stl1": "ts",
        "ws": "ws",
        "ssrd": "rsds",
        "strd": "rlds",
    },
    "training_indep_vars": ["tas", "snd", "rsds", "rlds", "ws"],
    "cmip_indep_vars": ["tas", "snd", "uas", "vas", "rsds", "rlds"],
    "cmip_dep_var": "ts",
    "training_time_tag": False,
    "training_overwrite": False,
    "projection_overwrite": False,
    "project_path": "projects/",
    "downloads_data_path": "downloads/",
    "era_data_path": "downloads/era5/",
    "cmip_data_path": "downloads/cmip/",
    "era_export_path": "inputs/era5/",
    "cmip_export_path": "inputs/cmip/",
    "training_data_path": "inputs/era5/",
    "projection_inputs_path": "inputs/cmip/",
    "models_path": "models/",
    "outputs_path": "outputs/",
}

model_settings = {
    "outliner_sd": 6,
    "test_set_proportion": 0.25,
    "random_seed": 777,
    "sequence_len": 7,
    "hidden_dim": 50,
    "num_layers": 1,
    "batch_size": 50,
    "num_epochs": 100,
}

# Uncomment to run each part:

# era_fetch.main(project_settings)
# era_preprocess.main(project_settings)
# training.main(project_settings, model_settings)
# cmip_canrcm_preprocess.main(project_settings)
# projection.main(project_settings, model_settings)
