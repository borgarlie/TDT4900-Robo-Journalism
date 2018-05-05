import random

from torch.distributions import Categorical

from models.GAN.generator_base import GeneratorBase
from utils.data_prep import *
from utils.logger import *
import time


class GeneratorSeqGanStrat(GeneratorBase):
    def __init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 batch_size, use_cuda, beta, num_monte_carlo_samples, sample_rate, negative_reward, use_trigram_check,
                 use_running_avg_baseline, discriminator_batch_size):
        GeneratorBase.__init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 batch_size, use_cuda, beta, num_monte_carlo_samples, sample_rate, negative_reward, use_trigram_check,
                 use_running_avg_baseline, discriminator_batch_size)

    # discriminator is used to calculate reward
    # target batch is used for MLE
    def train_on_batch(self, input_variable_batch, full_input_variable_batch, input_lengths, full_target_variable_batch,
                       target_lengths, discriminator, max_monte_carlo_length, target_variable, extended_vocabs,
                       full_target_variable_batch_2):

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        max_target_length = max(target_lengths)

        init_encoder_time_start = time.time()
        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        timings[timings_var_init_encoder] += (time.time() - init_encoder_time_start)

        # Argmax baseline
        if not self.use_running_avg_baseline:
            baseline = self.get_argmax_baseline(encoder_hidden, encoder_outputs, max_target_length,
                                                full_input_variable_batch, discriminator, full_target_variable_batch_2,
                                                extended_vocabs)
            print_baseline = baseline.sum(0) / self.batch_size

        # MLE loss
        if self.beta < 1.00:
            mle_loss = self.get_teacher_forcing_mle(encoder_hidden, encoder_outputs, max_target_length,
                                                    full_input_variable_batch, full_target_variable_batch,
                                                    target_variable)

        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        full_policy_values = []
        full_sequence_rewards = []
        accumulated_sequence = None

        policy_iteration_time_start = time.time()

        monte_carlo_length = min(max_target_length, max_monte_carlo_length)
        num_samples = 0

        eos_check_start = monte_carlo_length - 10

        multiply_time_2 = time.time()
        encoder_outputs_temp = multiply_data_in_dim(encoder_outputs, self.num_monte_carlo_samples, dim=1)
        full_input_variable_batch_temp = multiply_data_in_dim(full_input_variable_batch,
                                                              self.num_monte_carlo_samples, dim=0)
        timings[timings_var_copy_params] += time.time() - multiply_time_2

        # Policy iteration
        for di in range(monte_carlo_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)

            # Sample things
            # Currently always sampling the first token (to make sure there is at least 1 sampling per batch)
            sampling = True if random.random() <= self.sample_rate else False
            if sampling or di == 0:
                num_samples += 1
                m = Categorical(decoder_output)
                action = m.sample()
                log_prob = m.log_prob(action)
                full_policy_values.append(log_prob)

                monte_carlo_time_start = time.time()
                # Multiply the batch size by monte_carlo_samples
                multiply_time = time.time()
                # NOTE: Not sure if we need to do .clone() here, but just to be safe.
                action_temp = multiply_data_in_dim(action.clone(), self.num_monte_carlo_samples, dim=0)
                decoder_hidden_temp = multiply_data_in_dim(decoder_hidden, self.num_monte_carlo_samples, dim=1)
                if di == 0:
                    accumulated_sequence_temp = None
                else:
                    accumulated_sequence_temp = multiply_data_in_dim(accumulated_sequence, self.num_monte_carlo_samples,
                                                                     dim=0)
                batch_size_temp = self.batch_size * self.num_monte_carlo_samples
                timings[timings_var_copy_params] += time.time() - multiply_time

                sample_multiplied \
                    = self.monte_carlo_expansion(action_temp, decoder_hidden_temp, encoder_outputs_temp,
                                                 full_input_variable_batch_temp, accumulated_sequence_temp,
                                                 monte_carlo_length, batch_size_temp)

                monte_carlo_outer_time_start = time.time()
                current_reward = discriminator.evaluate(sample_multiplied, None, None)
                current_reward_chunked = current_reward.chunk(self.num_monte_carlo_samples, dim=0)
                temp_reward = 0
                for i in range(0, self.num_monte_carlo_samples):
                    temp_reward += current_reward_chunked[i]

                # calculate average reward
                avg_reward = temp_reward / self.num_monte_carlo_samples
                full_sequence_rewards.append(avg_reward)

                # add cumulative reward to calculate running average baseline
                if self.use_running_avg_baseline:
                    self.cumulative_reward += avg_reward.sum(0) / self.batch_size
                    self.updates += 1
                timings[timings_var_monte_carlo_outer] += (time.time() - monte_carlo_outer_time_start)
                timings[timings_var_monte_carlo] += (time.time() - monte_carlo_time_start)

            # Get top1 for the next input to the decoder
            topv, topi = decoder_output.data.topk(1)
            ni = topi

            # Check for EOS so that we stop sampling if all are EOS or PAD
            if di > eos_check_start:
                if is_whole_batch_pad_or_eos(ni):
                    break

            # Remove UNK before setting next input to decoder
            unk_check_time_start = time.time()
            ni = ni.squeeze(1)
            ni = where(ni < self.UPPER_BOUND, ni, self.MASK)
            decoder_input = Variable(ni.unsqueeze(1))
            timings[timings_var_unk_check] += time.time() - unk_check_time_start

            if accumulated_sequence is None:
                accumulated_sequence = decoder_input
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, decoder_input), 1)

        if self.use_running_avg_baseline:
            # Calculate running average baseline
            baseline = self.cumulative_reward / self.updates
            print_baseline = baseline.data[0]

        policy_loss = 0
        total_print_reward = 0
        total_print_adjusted_reward = 0

        print_log_sum = 0
        for i in range(0, len(full_policy_values)):
            print_log_sum += torch.sum(full_policy_values[i])
            total_print_reward += torch.sum(full_sequence_rewards[i])
            adjusted_full_sequence_reward = full_sequence_rewards[i] - baseline
            if not self.allow_negative_rewards:
                for j in range(0, len(adjusted_full_sequence_reward.data)):
                    if adjusted_full_sequence_reward.data[j] < 0.0:
                        adjusted_full_sequence_reward.data[j] = 0.0
            total_print_adjusted_reward += torch.sum(adjusted_full_sequence_reward)
            loss = -full_policy_values[i] * adjusted_full_sequence_reward
            policy_loss += torch.sum(loss) / self.batch_size
        print_log_sum = print_log_sum / self.batch_size
        total_print_reward = total_print_reward / self.batch_size
        total_print_reward = total_print_reward / num_samples
        total_print_adjusted_reward = total_print_adjusted_reward / self.batch_size
        total_print_adjusted_reward = total_print_adjusted_reward / num_samples

        timings[timings_var_policy_iteration] += (time.time() - policy_iteration_time_start)

        backprop_time_start = time.time()

        # divide by sequence length
        policy_loss = policy_loss / num_samples
        if self.beta < 1.00:
            mle_loss = mle_loss / max_target_length
            total_loss = self.beta * policy_loss + (1 - self.beta) * mle_loss
        else:
            total_loss = policy_loss

        total_loss.backward()

        clip = 2
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        timings[timings_var_backprop] += (time.time() - backprop_time_start)

        if self.beta < 1.00:
            return total_loss.data[0], mle_loss.data[0], policy_loss.data[0], print_log_sum.data[0], \
                   total_print_reward.data[0], print_baseline, total_print_adjusted_reward.data[0]
        else:
            return total_loss.data[0], total_loss.data[0], policy_loss.data[0], print_log_sum.data[0], \
                   total_print_reward.data[0], print_baseline, total_print_adjusted_reward.data[0]

    def monte_carlo_expansion(self, action, decoder_hidden, encoder_outputs, full_input_variable_batch,
                              initial_sequence, max_sample_length, batch_size):
        # Prepare next decoder input with UNK
        unk_check_time_start = time.time()
        action = action.data
        action = where(action < self.MONTE_CARLO_UPPER_BOUND, action, self.MONTE_CARLO_MASK)
        decoder_input = Variable(action.unsqueeze(1))
        timings[timings_var_unk_check] += time.time() - unk_check_time_start

        monte_carlo_cat_time_start_2 = time.time()
        if initial_sequence is not None:
            start = len(initial_sequence.data[0]) + 1
            decoder_output_variables = torch.cat((initial_sequence, decoder_input), 1)
        else:
            start = 1
            decoder_output_variables = decoder_input
        timings[timings_var_monte_carlo_cat] += (time.time() - monte_carlo_cat_time_start_2)

        for di in range(start, max_sample_length):
            monte_carlo_inner_time_start = time.time()
            decoder_output, decoder_hidden, _ \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               batch_size)
            timings[timings_var_monte_carlo_inner] += (time.time() - monte_carlo_inner_time_start)

            before_topk_monte = time.time()
            m = Categorical(decoder_output)
            action = m.sample()
            timings[timings_var_monte_carlo_top1] += (time.time() - before_topk_monte)

            unk_check_time_start = time.time()
            action = action.data
            action = where(action < self.MONTE_CARLO_UPPER_BOUND, action, self.MONTE_CARLO_MASK)
            decoder_input = Variable(action.unsqueeze(1))
            timings[timings_var_unk_check] += time.time() - unk_check_time_start

            monte_carlo_cat_time_start = time.time()
            decoder_output_variables = torch.cat((decoder_output_variables, decoder_input), 1)
            timings[timings_var_monte_carlo_cat] += (time.time() - monte_carlo_cat_time_start)

        # Add EOS to the samples that did not produce EOS, and add PAD to rest
        unk_check_time_start = time.time()
        last_tokens = where(action > self.EOS_MATRIX_MONTE_CARLO, self.EOS_MATRIX_MONTE_CARLO,
                            self.PAD_MATRIX_MONTE_CARLO)
        timings[timings_var_unk_check] += time.time() - unk_check_time_start
        monte_carlo_cat_time_start = time.time()
        decoder_output_variables = torch.cat((decoder_output_variables, Variable(last_tokens.unsqueeze(1))), 1)
        timings[timings_var_monte_carlo_cat] += (time.time() - monte_carlo_cat_time_start)

        return decoder_output_variables
