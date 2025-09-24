import os, logging

def set_logging_filehandler(
    log_file_path: str,
    mode: str = 'w',
    encoding: str = 'utf-8',
) -> None:
    '''
    Set logging FileHandler + StreamHandler

    Parameters:
        log_file_path: The directory + filename of log file. The os library
                       will make the directory where the file will be saved.
        mode:          The mode that specifies how the log file is opened.
                       Default 'w'.
        encoding:      The name of encoding that is used to encode or decode 
                       file. Default 'utf-8'.
    '''
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