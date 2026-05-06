##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import json
import numpy as np


class Parameter:
    def __init__(self, data, *args, **kwargs):
        if isinstance(data, Parameter):
            data = data.to_dict()

        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary or a Parameter object.")

        self._param_keys = set()
        for key, value in data.items():
            self.add_parameter(key, value)

    def __setattr__(self, key, value):
        """ Recursively convert nested dictionaries into Parameter objects. """

        if key.startswith('_'):
            super().__setattr__(key, value)
            return

        if hasattr(self, "_param_keys") and key in self._param_keys:
            value = self._force_json(value)
        super().__setattr__(key, value)

    def add_parameter(self, key, value):
        """ Add a new Parameter attribute. """

        self._param_keys.add(key)
        self.__setattr__(key, value)

    def _force_json(self, value):
        """ Convert value into a JSON-serializable object. """

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, dict):
            return Parameter(value)

        if isinstance(value, (list, tuple)):
            return [self._force_json(element) for element in value]

        if value is not None and not isinstance(value, (str, int, float, bool, Parameter)):
            raise TypeError(f"Object of type {type(value).__name__} is not JSON compatible!")
        return value

    def to_dict(self):
        """ Recursively export the object back to a standard dictionary. """

        result = {}
        for key in sorted(self._param_keys):
            value = getattr(self, key)
            if isinstance(value, Parameter):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [i.to_dict() if isinstance(i, Parameter) else i for i in value]
            else:
                result[key] = value
        return result

    def write(self, path):
        """ Stores the Parameter object in a pretty-printed JSON file. """

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False, sort_keys=True)

    @classmethod
    def read(cls, file_path, *args, **kwargs):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data, *args, **kwargs)

    def __str__(self):
        if hasattr(self, "model"):
            name = self.model
        else:
            name = ", ".join(sorted(self._param_keys))
        return f"{self.__class__.__name__}({name})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.to_dict()})"