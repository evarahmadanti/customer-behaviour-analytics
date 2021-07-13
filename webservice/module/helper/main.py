from configparser import ConfigParser
from pathlib import Path


class ConfigHelper:
    @staticmethod
    def get_file_config(filename, section):
        config = {}
        parser = ConfigParser()
        parser.read(Path(filename))
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                config[param[0]] = param[1]
        else:
            raise Exception(f'Section {section} not found in the {filename} file')
        return config
