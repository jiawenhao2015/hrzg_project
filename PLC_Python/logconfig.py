import logging
from datetime import datetime
import os
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# set two handlers
log_file = "logs/{}.log".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
# rm_file(log_file)
cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
# fileHandler = logging.FileHandler(os.path.join(cur_dir, log_file), mode = 'w')
fileHandler = RotatingFileHandler(os.path.join(cur_dir, log_file), mode = 'w',maxBytes=100*1024*1024, backupCount=5)


fileHandler.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)

# set formatter
formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)

# add
logger.addHandler(fileHandler)
logger.addHandler(consoleHandler)

