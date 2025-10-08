import os, logging
from datetime import datetime

def set_logging_filehandler(
    log_file_path: str,
    mode: str = 'w',
    encoding: str = 'utf-8',
) -> None:
    '''
    Set logging FileHandler + StreamHandler
    '''
    now = datetime.strftime(datetime.now(), format="%y%m%d_%H%M%S")
    log_file_path = log_file_path + f'_{now}.log'
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    format = '%(asctime)s %(message)s'
    logging.basicConfig(
        format=format,
        handlers=[
            logging.FileHandler(
                log_file_path,
                mode=mode,
                encoding=encoding,
            ),
            logging.StreamHandler(),
        ],
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )
    logging.info(f'Logging (File + Stream) Initialized')

def set_logging_streamhandler() -> None:
    '''
    Set logging StreamHandler
    '''
    logging.basicConfig(
        format=format,
        handlers=[
            logging.StreamHandler(),
        ],
        datefmt='%H:%M:%S',
        level=logging.INFO,
    )
    logging.info(f'Logging (Stream) Initialized')