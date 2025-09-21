import logging

def function(script_name):
    logger=logging.getLogger(script_name)
    logger.setLevel(logging.DEBUG)
    handler=logging.FileHandler(f'C:\\Users\\nikhi\\Downloads\\Credit Card Project\\log_files\\{script_name}.log',mode='w')
    formatter=logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
