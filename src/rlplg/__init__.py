__author__ = "guilherme"
__version__ = "0.10.2"
__email__ = "guilherme@dsv.su.se"
__description__ = "RL-Playground"
__uri__ = "https://github.com/guidj/rlplg"

import logging.config
import os
import os.path

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"},
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        "loggers": {"": {"handlers": ["default"], "level": "INFO", "propagate": True}},
    }
)

SRC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
