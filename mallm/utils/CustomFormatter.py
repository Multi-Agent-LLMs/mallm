# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
import logging
from colorama import Fore


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: Fore.BLUE + format,
        logging.INFO: Fore.WHITE + format,
        logging.WARNING: Fore.YELLOW + format,
        logging.ERROR: Fore.LIGHTRED_EX + format,
        logging.CRITICAL: Fore.RED + format,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
