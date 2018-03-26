import random

from evaluation.seq2seq.evaluate import calculate_loss_on_eval_set, evaluate_argmax
from training.seq2seq.train import save_state
from utils.batching import *
from utils.data_prep import *
from utils.time_utils import *
from utils.logger import *


def train_GAN(config, generator, discriminator, training_pairs, eval_pairs, max_article_length, max_abstract_length,
              writer):

    log_message("Starting GAN training")
    n_generator = config['train']['n_generator']
    n_discriminator = config['train']['n_discriminator']  # This is a scaling factor of n_generator
    discriminator_n_epochs = config['train']['discriminator_n_epochs']
    max_sample_length = config['train']['max_sample_length']
    if max_sample_length > max_abstract_length:
        max_sample_length = max_abstract_length
    n_epochs = config['train']['n_epochs']
    batch_size = config['train']['batch_size']
    print_every = config['log']['print_every']

    start = time.time()
    print_loss_generator = 0
    print_loss_mle = 0
    print_loss_policy = 0
    print_loss_discriminator = 0
    print_total_reward = 0
    print_adjusted_reward = 0
    lowest_loss_generator = 999
    lowest_loss_discriminator = 999

    itr_discriminator = 0
    itr_generator = 0

    num_batches = int(len(training_pairs) / batch_size)
    n_iters = num_batches * n_epochs
    total_runtime = 0

    # generate ground truth to use for discriminator (always the same number of positive and negative samples)
    ground_truth = [1 for _ in range(batch_size)] + [0 for _ in range(batch_size)]
    ground_truth_batched = Variable(torch.FloatTensor(ground_truth)).unsqueeze(1)
    if use_cuda:
        ground_truth_batched = ground_truth_batched.cuda()

    # train GAN for n_epochs
    for epoch in range(1, n_epochs+1):
        log_message("Starting epoch %d" % epoch)
        # shuffle articles
        random.shuffle(training_pairs)

        chunking_time_start = time.time()

        # split into batches
        training_batches = list(chunks(training_pairs, batch_size))

        timings[timings_var_chunkings] += (time.time() - chunking_time_start)

        count_disc = 0
        batch = 0
        while batch < num_batches:
            # train generator for n_generator batches
            for n in range(n_generator):
                init_generator_time_start = time.time()

                input_variable, full_input_variable, input_lengths, _, full_target_var, target_lengths \
                    = prepare_batch(batch_size, training_batches[batch], max_article_length, max_abstract_length)

                timings[timings_var_init_generator] += (time.time() - init_generator_time_start)

                generator_train_time_start = time.time()

                loss, mle_loss, policy_loss, reward, adjusted_reward = generator.train_on_batch(
                    input_variable, full_input_variable, input_lengths, full_target_var, target_lengths, discriminator,
                    max_sample_length)

                timings[timings_var_generator_train] += (time.time() - generator_train_time_start)

                print_loss_generator += loss
                print_loss_mle += mle_loss
                print_loss_policy += policy_loss
                print_total_reward += reward
                print_adjusted_reward += adjusted_reward
                # calculate number of batches processed
                itr_generator += 1
                if itr_generator % print_every == 0:
                    print_loss_avg = print_loss_generator / print_every
                    print_loss_avg_mle = print_loss_mle / print_every
                    print_loss_avg_policy = print_loss_policy / print_every
                    print_total_reward_avg = print_total_reward / print_every
                    print_adjusted_reward_avg = print_adjusted_reward / print_every
                    print_loss_generator = 0
                    print_loss_mle = 0
                    print_loss_policy = 0
                    print_total_reward = 0
                    print_adjusted_reward = 0
                    progress, total_runtime = time_since(start, itr_generator / n_iters, total_runtime)
                    start = time.time()
                    log_message('%s (%d %d%%)' % (progress, itr_generator, itr_generator / n_iters * 100))
                    log_message('Generator loss (total, mle, policy, reward, adjusted_reward): %.4f, %.4f, %.4f, %.4f, '
                                '%.4f' % (print_loss_avg, print_loss_avg_mle, print_loss_avg_policy,
                                          print_total_reward_avg, print_adjusted_reward_avg))
                    if print_loss_avg < lowest_loss_generator:
                        lowest_loss_generator = print_loss_avg
                        log_message(" ^ Lowest generator loss so far")
                    log_profiling(print_every, n_discriminator)
                    # Generating a few arg max summaries to see if there are differences
                    generator.encoder.eval()
                    generator.decoder.eval()
                    samples = eval_pairs[0:2]
                    evaluate_argmax(generator.vocabulary, samples, generator.encoder, generator.decoder,
                                    max_abstract_length)
                    generator.encoder.train()
                    generator.decoder.train()
                batch += 1
                # generate n_discriminator batches to train discriminator on
                discriminator_training_data = []
                for m in range(n_discriminator):
                    # generate fake data
                    # pad_abstract_length = max_sample_length

                    init_descriminator_time_start = time.time()

                    pad_abstract_length = max_abstract_length
                    # TODO: Check if we need to set it to 101(?) or if we can set it lower (i.e. max_sample_length)

                    real_data_article_variable, full_real_data_article_variable, real_data_article_lengths, \
                    real_data_variable, _, _ = prepare_batch(batch_size, training_batches[count_disc],
                                                             max_article_length, pad_abstract_length)

                    real_data_variable = real_data_variable.transpose(1, 0)

                    create_fake_time_start = time.time()

                    fake_data_variable = generator.create_samples(
                        real_data_article_variable, full_real_data_article_variable, real_data_article_lengths,
                        max_sample_length, pad_abstract_length)

                    timings[timings_var_create_fake] += (time.time() - create_fake_time_start)

                    d_titles_real_and_fake = torch.cat((real_data_variable, fake_data_variable), 0)
                    discriminator_training_data.append(d_titles_real_and_fake)
                    count_disc += 1
                    count_disc = count_disc % len(training_batches)

                    timings[timings_var_init_discriminator] += (time.time() - init_descriminator_time_start)

                discriminator_train_time_start = time.time()

                # train discriminator for discriminator_n_epochs epochs
                for k in range(discriminator_n_epochs):
                    # train discriminator on all sample batches
                    for m in range(n_discriminator):
                        # train and calculate loss
                        loss = discriminator.train(ground_truth_batched, discriminator_training_data[m])
                        print_loss_discriminator += loss
                        # calculate number of batches processed
                        # itr_discriminator = (epoch - 1) * num_batches + batch + (k) * n_discriminator + m
                        itr_discriminator += 1
                        if itr_discriminator % print_every == 0:
                            print_loss_avg = print_loss_discriminator / print_every
                            print_loss_discriminator = 0
                            log_message('Discriminator loss at %d - %d - %d - %d - : %.4f'
                                  % (itr_discriminator, batch, k, m, print_loss_avg))
                            if print_loss_avg < lowest_loss_discriminator:
                                lowest_loss_discriminator = print_loss_avg
                                log_message(" ^ Lowest discriminator loss so far")

                timings[timings_var_discriminator_train] += (time.time() - discriminator_train_time_start)

            # update generator beta - the parameters of the sampling model is now freezed until next round
            copy_params_time_start = time.time()

            generator.update_generator_beta_params()

            timings[timings_var_copy_params] += (time.time() - copy_params_time_start)

        # save each epoch
        log_message("Saving model")
        save_state({
            'model_state_encoder': generator.encoder.state_dict(),
            'model_state_decoder': generator.decoder.state_dict(),
        }, config['experiment_path'] + "/" + config['save']['save_file_generator'])
        save_state({
            'model': discriminator.model.state_dict()
        }, config['experiment_path'] + "/" + config['save']['save_file_discriminator'])

        generator.encoder.eval()
        generator.decoder.eval()
        calculate_loss_on_eval_set(config, generator.vocabulary, generator.encoder, generator.decoder,
                                   generator.mle_criterion, writer, epoch, max_article_length, eval_pairs,
                                   use_logger=False)
        generator.encoder.train()
        generator.decoder.train()
