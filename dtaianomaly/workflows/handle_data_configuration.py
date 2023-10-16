
import json
from typing import Dict, Any, Union
from dtaianomaly.data_management import DataManager

DataConfiguration = Union[Dict[str, Any], str]


def handle_data_configuration(data_manager: DataManager, data_configuration: DataConfiguration) -> DataManager:

    # Read the data configuration file if it is a string
    if type(data_configuration) is str:
        configuration_file = open(data_configuration, 'r')
        data_configuration = json.load(configuration_file)
        configuration_file.close()

    # Select the datasets that match the given properties
    for select_options in data_configuration['select']:
        data_manager.select(select_options)

    # Make sure that all the selected datasets are available
    data_manager.filter_available_datasets()

    # Return the data manager
    return data_manager
