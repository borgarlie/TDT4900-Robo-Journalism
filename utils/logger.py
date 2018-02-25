import logging

logger = logging.getLogger()


def init_logger(filename):
    # create logger
    logging.basicConfig(filename=filename, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # add ch to logger
    logger.addHandler(ch)
    # return logger


def log_message(message):
    logger.info(message)


def log_error_message(message):
    logger.error(message)


def log_training_message(progress, itr, percentage, print_loss_avg, lowest_loss):
    logger.info('%s (%d %d%%) %.4f' % (progress, itr, percentage, print_loss_avg))
    if print_loss_avg < lowest_loss:
        lowest_loss = print_loss_avg
        logger.info(" ^ Lowest loss so far")
    return lowest_loss
