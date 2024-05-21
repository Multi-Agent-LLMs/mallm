# https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
import logging

from colorama import Fore


class CustomFormatter(logging.Formatter):
    # grey = "\x1b[38;20m"
    # yellow = "\x1b[33;20m"
    # red = "\x1b[31;20m"
    # bold_red = "\x1b[31;1m"
    # reset = "\x1b[0m"
    format_str = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: Fore.BLUE + format_str,
        logging.INFO: Fore.WHITE + format_str,
        logging.WARNING: Fore.YELLOW + format_str,
        logging.ERROR: Fore.LIGHTRED_EX + format_str,
        logging.CRITICAL: Fore.RED + format_str,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
