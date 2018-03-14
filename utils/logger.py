from utils.time_utils import as_minutes
import json
import logging

logger = logging.getLogger()


# Random things
timings_var_chunkings = 'CHUNKING'
# Generator timings
timings_var_init_generator = 'INIT_GENERATOR_VARS'
timings_var_generator_train = 'GENERATOR_TRAIN'
timings_var_init_encoder = 'INIT_ENCODER'
timings_var_baseline = 'BASELINE'
timings_var_baseline_inner = 'BASELINE_INNER'
timings_var_monte_carlo = 'MONTE_CARLO'
timings_var_monte_carlo_outer = 'MONTE_CARLO_OUTER'
timings_var_monte_carlo_cat = 'MONTE_CARLO_CAT'
timings_var_monte_carlo_encoder = 'MONTE_CARLO_ENCODER'
timings_var_monte_carlo_inner = 'MONTE_CARLO_INNER'
timings_var_monte_carlo_inner_transpose = 'MONTE_CARLO_INNER_TRANSPOSE'
timings_var_backprop = 'BACKPROP'
timings_var_copy_params = 'COPY_PARAMS'
timings_var_policy_iteration = 'POLICY_ITERATION'


# Discriminator timings
timings_var_init_discriminator = 'INIT_DISCRIMINATOR_VARS'
timings_var_create_fake = 'CREATE_FAKE'
timings_var_create_fake_inner = 'CREATE_FAKE_INNER'
timings_var_discriminator_train = 'DISCRIMINATOR_TRAIN'

timings = {}
timings[timings_var_chunkings] = 0.0
timings[timings_var_init_generator] = 0.0
timings[timings_var_generator_train] = 0.0
timings[timings_var_init_encoder] = 0.0
timings[timings_var_baseline] = 0.0
timings[timings_var_baseline_inner] = 0.0
timings[timings_var_monte_carlo] = 0.0
timings[timings_var_monte_carlo_outer] = 0.0
timings[timings_var_monte_carlo_cat] = 0.0
timings[timings_var_monte_carlo_encoder] = 0.0
timings[timings_var_monte_carlo_inner] = 0.0
timings[timings_var_monte_carlo_inner_transpose] = 0.0
timings[timings_var_backprop] = 0.0
timings[timings_var_copy_params] = 0.0
timings[timings_var_policy_iteration] = 0.0
timings[timings_var_init_discriminator] = 0.0
timings[timings_var_create_fake] = 0.0
timings[timings_var_create_fake_inner] = 0.0
timings[timings_var_discriminator_train] = 0.0

decode_breaking_monte_carlo_sampling = 'MONTE_CARLO'
decode_breaking_baseline = 'BASELINE'
decode_breaking_policy = 'POLICY'
decode_breaking_fake_sampling = 'FAKE_SAMPLING'

monte_carlo_sampling_num = 'NUM_SAMPLES'
monte_carlo_sampling = {}
monte_carlo_sampling[decode_breaking_monte_carlo_sampling] = 0
monte_carlo_sampling[monte_carlo_sampling_num] = 0

decode_breakings = {}
decode_breakings[decode_breaking_baseline] = 0
decode_breakings[decode_breaking_policy] = 0
decode_breakings[decode_breaking_fake_sampling] = 0


def init_logger(filename):
    # create logger
    logging.basicConfig(filename=filename, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)

    # add ch to logger
    # logger.addHandler(ch)
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


def log_profiling(num_iterations):
    log_timings()
    log_decode_breakings(num_iterations)
    log_monte_carlo_sampling()


def log_timings():
    # calculate minutes for each timing
    for var in timings:
        temp_minutes = as_minutes(timings[var])
        timings[var] = temp_minutes

    logger.info(json.dumps(timings, indent=2))
    # Reset the timings
    for var in timings:
        timings[var] = 0.0


def log_decode_breakings(num_iterations):
    for var in decode_breakings:
        avg = decode_breakings[var] / num_iterations
        decode_breakings[var] = avg

    logger.info(json.dumps(decode_breakings, indent=2))

    for var in decode_breakings:
        decode_breakings[var] = 0


def log_monte_carlo_sampling():
    avg = monte_carlo_sampling[decode_breaking_monte_carlo_sampling] / monte_carlo_sampling[monte_carlo_sampling_num]
    logger.info("Monte carlo sampling average breaking: %.1f" % avg)
    monte_carlo_sampling[decode_breaking_monte_carlo_sampling] = 0
    monte_carlo_sampling[monte_carlo_sampling_num] = 0
