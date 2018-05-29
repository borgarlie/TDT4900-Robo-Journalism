import random

from evaluation.seq2seq.evaluate import calculate_loss_on_eval_set, evaluate_argmax, evaluate_rouge
from training.seq2seq.train import save_state
from utils.batching import *
from utils.data_prep import *
from utils.time_utils import *
from utils.logger import *


def train_GAN(config, generator, discriminator, training_pairs, eval_pairs, max_article_length, max_abstract_length,
              writer):

    log_message("Starting GAN training")
    n_generator = config['train']['n_generator']
    n_discriminator = config['train']['n_discriminator']
    discriminator_n_epochs = config['train']['discriminator_n_epochs']
    discriminator_fake_data_sample_rate = config['train']['discriminator_fake_data_sample_rate']
    max_sample_length = config['train']['max_sample_length']
    if max_sample_length > max_abstract_length:
        max_sample_length = max_abstract_length
    n_epochs = config['train']['n_epochs']
    batch_size = config['train']['batch_size']
    discriminator_batch_size = config['train']['discriminator_batch_size']
    print_every = config['log']['print_every']

    start = time.time()
    print_loss_generator = 0
    print_loss_mle = 0
    print_loss_policy = 0
    print_loss_policy_log_sum = 0
    print_loss_discriminator = 0
    print_total_reward = 0
    print_total_gan_reward = 0
    print_total_rouge_reward = 0
    print_baseline = 0
    print_adjusted_reward = 0
    lowest_loss_generator = 999
    lowest_loss_discriminator = 999

    # Print first 3 argmax before training to compare
    # generator.encoder.eval()
    # generator.decoder.eval()
    # samples = eval_pairs[0:3]
    # evaluate_argmax(generator.vocabulary, samples, generator.encoder, generator.decoder,
    #                 max_abstract_length)
    # rouge_samples = eval_pairs[0:1000]
    # rouge_score = evaluate_rouge(rouge_samples, generator.encoder, generator.decoder, max_article_length,
    #                              max_abstract_length, discriminator)
    # log_message("Rouge score before training: %.6f" % rouge_score)
    # generator.encoder.train()
    # generator.decoder.train()

    itr_discriminator = 0
    itr_generator = 0

    num_batches = int(len(training_pairs) / batch_size)
    n_iters = num_batches * n_epochs
    total_runtime = 0

    # generate ground truth to use for discriminator (always the same number of positive and negative samples)
    ground_truth = [[1, 0] for _ in range(discriminator_batch_size)] + [[0, 1] for _ in range(discriminator_batch_size)]
    ground_truth_batched = Variable(torch.FloatTensor(ground_truth))
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
        # TODO: Could it be useful to re-shuffle for these batches?
        # a seperate list for discriminator batches because of different batch size
        if batch_size == discriminator_batch_size:
            discriminator_training_batches = training_batches
        else:
            discriminator_training_batches = list(chunks(training_pairs, discriminator_batch_size))[:-1]
        timings[timings_var_chunkings] += (time.time() - chunking_time_start)

        count_disc = 0
        batch = 0
        while batch < num_batches:
            # train generator for n_generator batches
            for n in range(n_generator):
                init_generator_time_start = time.time()
                input_variable, full_input_variable, input_lengths, target_var, full_target_var, target_lengths, extended_vocabs, full_target_var_2 \
                    = prepare_batch(batch_size, training_batches[batch], max_article_length, max_abstract_length)
                timings[timings_var_init_generator] += (time.time() - init_generator_time_start)

                generator_train_time_start = time.time()
                loss, mle_loss, policy_loss, policy_log_sum, reward, baseline, adjusted_reward, gan_reward, \
                rouge_reward = generator.train_on_batch(input_variable, full_input_variable, input_lengths,
                                                full_target_var, target_lengths, discriminator, max_sample_length,
                                                target_var, extended_vocabs, full_target_var_2)
                timings[timings_var_generator_train] += (time.time() - generator_train_time_start)

                print_loss_generator += loss
                print_loss_mle += mle_loss
                print_loss_policy += policy_loss
                print_loss_policy_log_sum += policy_log_sum
                print_total_reward += reward
                print_baseline += baseline
                print_adjusted_reward += adjusted_reward
                print_total_gan_reward += gan_reward
                print_total_rouge_reward += rouge_reward
                # calculate number of batches processed
                itr_generator += 1
                if itr_generator % print_every == 0:
                    print_loss_avg = print_loss_generator / print_every
                    print_loss_avg_mle = print_loss_mle / print_every
                    print_loss_avg_policy = print_loss_policy / print_every
                    print_loss_avg_policy_log_sum = print_loss_policy_log_sum / print_every
                    print_total_reward_avg = print_total_reward / print_every
                    print_avg_baseline = print_baseline / print_every
                    print_adjusted_reward_avg = print_adjusted_reward / print_every
                    print_disc_loss_avg = print_loss_discriminator / print_every
                    print_gan_reward_avg = print_total_gan_reward / print_every
                    print_rouge_reward_avg = print_total_rouge_reward / print_every
                    print_loss_discriminator = 0
                    print_loss_generator = 0
                    print_loss_mle = 0
                    print_loss_policy = 0
                    print_loss_policy_log_sum = 0
                    print_total_reward = 0
                    print_baseline = 0
                    print_adjusted_reward = 0
                    print_total_gan_reward = 0
                    print_total_rouge_reward = 0
                    progress, total_runtime = time_since(start, itr_generator / n_iters, total_runtime)
                    start = time.time()
                    log_message('Discriminator loss at %d: %.4f' % (itr_discriminator, print_disc_loss_avg))
                    log_message('%s (%d %d%%)' % (progress, itr_generator, itr_generator / n_iters * 100))
                    log_message('Generator loss (total, mle, policy, policy_log_sum, reward, baseline, adjusted_reward,'
                                ' gan_reward, rouge_reward)'
                                ': %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f'
                                % (print_loss_avg, print_loss_avg_mle, print_loss_avg_policy,
                                   print_loss_avg_policy_log_sum, print_total_reward_avg, print_avg_baseline,
                                   print_adjusted_reward_avg, print_gan_reward_avg, print_rouge_reward_avg))
                    # Making a loss dictionary for tensorboard
                    log_dict = {'total_loss': print_loss_avg, 'mle': print_loss_avg_mle,
                                'policy': print_loss_avg_policy, 'policy_log_sum': print_loss_avg_policy_log_sum,
                                'reward': print_total_reward_avg, 'baseline': print_avg_baseline,
                                'adjusted_reward': print_adjusted_reward_avg, 'disc_loss': print_disc_loss_avg,
                                'gan_reward': print_gan_reward_avg, 'rouge_reward': print_rouge_reward_avg}
                    writer.add_scalars('Logging', log_dict, itr_generator)
                    if print_loss_avg < lowest_loss_generator:
                        lowest_loss_generator = print_loss_avg
                        log_message(" ^ Lowest generator loss so far")
                    log_profiling(print_every, n_discriminator)
                    # Generating a few arg max summaries to see if there are differences
                    # generator.encoder.eval()
                    # generator.decoder.eval()
                    # samples = eval_pairs[0:3]
                    # evaluate_argmax(generator.vocabulary, samples, generator.encoder, generator.decoder,
                    #                 max_abstract_length)
                    # rouge_samples = eval_pairs[0:1000]
                    # rouge_score = evaluate_rouge(rouge_samples, generator.encoder, generator.decoder,
                    #                              max_article_length,
                    #                              max_abstract_length, discriminator)
                    # log_message("Rouge score: %.6f" % rouge_score)
                    # generator.encoder.train()
                    # generator.decoder.train()
                batch += 1
                # generate n_discriminator batches to train discriminator on
            discriminator_training_data = []
            for m in range(n_discriminator):
                # generate fake data
                # Alternate between sampling and argmax for fake data
                sample = True if random.random() <= discriminator_fake_data_sample_rate else False

                init_descriminator_time_start = time.time()
                pad_abstract_length = max_abstract_length
                real_data_article_variable, full_real_data_article_variable, real_data_article_lengths, \
                real_data_variable, _, target_lengths, _, _ \
                    = prepare_batch(discriminator_batch_size, discriminator_training_batches[count_disc],
                                    max_article_length, pad_abstract_length)
                real_data_variable = real_data_variable.transpose(1, 0)

                create_fake_time_start = time.time()
                fake_data_variable = generator.create_samples(
                    real_data_article_variable, full_real_data_article_variable, real_data_article_lengths,
                    pad_abstract_length, pad_abstract_length, discriminator_batch_size, sample=sample)

                timings[timings_var_create_fake] += (time.time() - create_fake_time_start)

                d_titles_real_and_fake = torch.cat((real_data_variable, fake_data_variable), 0)
                discriminator_training_data.append(d_titles_real_and_fake)
                count_disc += 1
                count_disc = count_disc % len(discriminator_training_batches)
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
                    itr_discriminator += 1
                    # if itr_discriminator % print_every == 0:
                    #     print_loss_avg = print_loss_discriminator / print_every
                    #     print_loss_discriminator = 0
                    #     log_message('Discriminator loss at %d - %d - %d - %d - : %.4f'
                    #                 % (itr_discriminator, batch, k, m, print_loss_avg))
                    #     if print_loss_avg < lowest_loss_discriminator:
                    #         lowest_loss_discriminator = print_loss_avg
                    #         log_message(" ^ Lowest discriminator loss so far")

            timings[timings_var_discriminator_train] += (time.time() - discriminator_train_time_start)

            # save on quarter epoch
            if batch == int(num_batches / 4):
                log_message("Saving model on half epoch")
                save_state({
                    'model_state_encoder': generator.encoder.state_dict(),
                    'model_state_decoder': generator.decoder.state_dict(),
                }, config['experiment_path'] + "/" + "/epoch%d_1quarter_" % epoch + config['save']['save_file_generator'])
            # save on half epoch
            if batch == int(num_batches / 2):
                log_message("Saving model on half epoch")
                save_state({
                    'model_state_encoder': generator.encoder.state_dict(),
                    'model_state_decoder': generator.decoder.state_dict(),
                }, config['experiment_path'] + "/" + "/epoch%d_2quarter_" % epoch + config['save']['save_file_generator'])
            # save on third quarter epoch
            if batch == int(int(num_batches / 4) * 3):
                log_message("Saving model on half epoch")
                save_state({
                    'model_state_encoder': generator.encoder.state_dict(),
                    'model_state_decoder': generator.decoder.state_dict(),
                }, config['experiment_path'] + "/" + "/epoch%d_3quarter_" % epoch + config['save']['save_file_generator'])

        # save each epoch
        log_message("Saving model")
        save_state({
            'model_state_encoder': generator.encoder.state_dict(),
            'model_state_decoder': generator.decoder.state_dict(),
        }, config['experiment_path'] + "/" + "/epoch%d_" % epoch + config['save']['save_file_generator'])
        # save_state({
        #     'model': discriminator.model.state_dict()
        # }, config['experiment_path'] + "/" + "/epoch%d_" % epoch + config['save']['save_file_discriminator'])

        generator.encoder.eval()
        generator.decoder.eval()
        calculate_loss_on_eval_set(config, generator.vocabulary, generator.encoder, generator.decoder,
                                   generator.mle_criterion, writer, epoch, max_article_length, eval_pairs,
                                   use_logger=False)
        generator.encoder.train()
        generator.decoder.train()
