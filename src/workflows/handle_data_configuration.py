
import json
from typing import Dict, Any, Union
from src.data_management import DataManager

DataConfigurationType = Union[Dict[str, Any], str]


def handle_data_configuration(data_manager: DataManager, data_configuration: DataConfigurationType) -> DataManager:

    # Read the data configuration file if it is a string
    if type(data_configuration) is str:
        configuration_file = open(data_configuration, 'r')
        data_configuration = json.load(configuration_file)
        configuration_file.close()

    # Select the datasets that match the given properties
    for select_options in data_configuration['select']:
        data_manager.select(select_options)

    # Return the data manager
    return data_manager
