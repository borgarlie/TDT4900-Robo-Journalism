import random

from torch.distributions import Categorical

from utils.data_prep import *
from utils.logger import *
import time


class GeneratorSuperStrat:
    def __init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 batch_size, use_cuda, beta, num_monte_carlo_samples, sample_rate, negative_reward):
        self.vocabulary = vocabulary
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.mle_criterion = mle_criterion
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.beta = beta
        self.num_monte_carlo_samples = num_monte_carlo_samples
        self.updates = 0
        self.cumulative_reward = 0.0
        self.sample_rate = sample_rate
        self.allow_negative_rewards = negative_reward
        self.use_trigram_check = False
        self.use_running_avg_baseline = False

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
            print_baseline = baseline.mean()

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
        policy_iteration_break_early = False

        start_check_for_pad_and_eos = int(max_target_length / 3) * 2

        monte_carlo_length = min(max_target_length, max_monte_carlo_length)
        num_samples = 0

        # Policy iteration
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)

            topv, topi = decoder_output.data.topk(1)
            ni = topi
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni).cuda()

            # Sample things
            # Currently always sampling the first token (to make sure there is at least 1 sampling per batch)
            sampling = True if random.random() <= self.sample_rate else False
            if sampling or di == 0:
                num_samples += 1
                m = Categorical(decoder_output)
                action = m.sample()
                log_prob = m.log_prob(action)
                full_policy_values.append(log_prob)

                for token_index in range(0, len(action)):
                    if action[token_index].data[0] >= self.vocabulary.n_words:
                        action[token_index].data[0] = UNK_token
                monte_carlo_input = action.unsqueeze(1)

                monte_carlo_time_start = time.time()
                sample = self.monte_carlo_expansion(monte_carlo_input, decoder_hidden, encoder_outputs,
                                                    full_input_variable_batch, accumulated_sequence, monte_carlo_length)
                monte_carlo_outer_time_start = time.time()
                current_reward = discriminator.evaluate(sample, full_target_variable_batch_2, extended_vocabs)
                full_sequence_rewards.append(current_reward)

                # add cumulative reward to calculate running average baseline
                if self.use_running_avg_baseline:
                    self.cumulative_reward += current_reward.mean().data[0]
                    self.updates += 1
                timings[timings_var_monte_carlo_outer] += (time.time() - monte_carlo_outer_time_start)
                timings[timings_var_monte_carlo] += (time.time() - monte_carlo_time_start)

            # Update accumulated sequence
            if accumulated_sequence is None:
                accumulated_sequence = decoder_input
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, decoder_input), 1)

            # Break the policy iteration loop if all the variables in the batch is at EOS or PAD
            if di > start_check_for_pad_and_eos:
                if is_whole_batch_pad_or_eos(decoder_input.data):
                    decode_breakings[decode_breaking_policy] += di
                    policy_iteration_break_early = True
                    break

        if not policy_iteration_break_early:
            decode_breakings[decode_breaking_policy] += max_target_length - 1

        if self.use_running_avg_baseline:
            # Calculate running average baseline
            avg = self.cumulative_reward / self.updates
            baseline = Variable(torch.cuda.FloatTensor([avg]))
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

        # TODO: MLE should probably be divided too ?
        # divide by sequence length
        policy_loss = policy_loss / num_samples

        if self.beta < 1.00:
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
                   total_print_reward, print_baseline, total_print_adjusted_reward
        else:
            return total_loss.data[0], total_loss.data[0], policy_loss.data[0], print_log_sum.data[0], \
                   total_print_reward, print_baseline, total_print_adjusted_reward

    def get_teacher_forcing_mle(self, encoder_hidden, encoder_outputs, max_target_length, full_input_variable_batch,
                                full_target_variable_batch, target_variable):
        mle_loss = 0
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            log_output = torch.log(decoder_output.clamp(min=1e-8))
            mle_loss += self.mle_criterion(log_output, full_target_variable_batch[di])
            decoder_input = target_variable[di]
        return mle_loss

    def get_argmax_baseline(self, encoder_hidden, encoder_outputs, max_target_length, full_input_variable_batch,
                            discriminator, full_target_variable_batch_2, extended_vocabs):
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        accumulated_sequence = None
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)

            if self.use_trigram_check:
                # Using topk > 1 to compensate for possible trigram overlaps
                topv, topi = decoder_output.data.topk(100)
                next_words = []
                for batch in range(0, self.batch_size):
                    topk = 0
                    while 1:
                        next_word = topi[batch][topk]
                        if di > 2 and has_trigram(accumulated_sequence[batch].data, next_word):
                            topk += 1
                            continue
                        break
                    next_words.append(next_word)
                ni = torch.cuda.LongTensor(next_words).unsqueeze(1)
            else:
                topv, topi = decoder_output.data.topk(1)
                ni = topi  # next input, batch of top softmax

            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)

            if accumulated_sequence is None:
                accumulated_sequence = decoder_input
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, decoder_input), 1)

        baseline = discriminator.evaluate(accumulated_sequence, full_target_variable_batch_2, extended_vocabs)
        return baseline

    def monte_carlo_expansion(self, sampled_token, decoder_hidden, encoder_outputs, full_input_variable_batch,
                              initial_sequence, max_sample_length):

        # TODO: Do we need to set eval mode here? Guess not?
        # Should we set require_grad = False? (Seems not)
        # for param in self.decoder.parameters():
        #     param.require_grad = False

        start_check_for_pad_and_eos = int(max_sample_length / 3) * 2

        if initial_sequence is not None:
            start = len(initial_sequence.data[0]) + 1
            decoder_output_variables = torch.cat((initial_sequence, sampled_token), 1)
        else:
            start = 1
            decoder_output_variables = sampled_token

        decoder_input = sampled_token

        monte_carlo_sampling_break_early = False
        for di in range(start, max_sample_length):
            monte_carlo_inner_time_start = time.time()
            decoder_output, decoder_hidden, _ \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            timings[timings_var_monte_carlo_inner] += (time.time() - monte_carlo_inner_time_start)

            before_topk_monte = time.time()
            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)
            timings[timings_var_monte_carlo_top1] += (time.time() - before_topk_monte)

            monte_carlo_cat_time_start = time.time()
            decoder_output_variables = torch.cat((decoder_output_variables, decoder_input), 1)
            timings[timings_var_monte_carlo_cat] += (time.time() - monte_carlo_cat_time_start)

            if di > start_check_for_pad_and_eos:
                if is_whole_batch_pad_or_eos(ni):
                    monte_carlo_sampling[decode_breaking_monte_carlo_sampling] += di
                    monte_carlo_sampling[monte_carlo_sampling_num] += 1
                    monte_carlo_sampling_break_early = True
                    break

        if not monte_carlo_sampling_break_early:
            monte_carlo_sampling[decode_breaking_monte_carlo_sampling] += max_sample_length - 1
            monte_carlo_sampling[monte_carlo_sampling_num] += 1

        return decoder_output_variables

    # Used to create fake data samples to train the discriminator
    # Returned values as batched sentences as variables
    def create_samples(self, input_variable_batch, full_input_variable_batch, input_lengths, max_sample_length,
                       pad_length):

        self.encoder.eval()
        self.decoder.eval()

        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        decoder_outputs = [[] for _ in range(0, self.batch_size)]
        create_fake_sample_break_early = False

        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_sample_length):

            create_fake_inner_time_start = time.time()
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            timings[timings_var_create_fake_inner] += (time.time() - create_fake_inner_time_start)

            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax scores
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)

            decoder_output_data = ni.cpu().numpy()
            for batch_index in range(0, len(decoder_output_data)):
                decoder_outputs[batch_index].append(decoder_output_data[batch_index].item())

            if is_whole_batch_pad_or_eos(ni):
                decode_breakings[decode_breaking_fake_sampling] += di
                create_fake_sample_break_early = True
                break

        if not create_fake_sample_break_early:
            decode_breakings[decode_breaking_fake_sampling] += max_sample_length - 1

        decoder_outputs_padded = [pad_seq(s, pad_length) for s in decoder_outputs]
        decoder_output_variables = Variable(torch.LongTensor(decoder_outputs_padded))
        if self.use_cuda:
            decoder_output_variables = decoder_output_variables.cuda()

        self.encoder.train()
        self.decoder.train()

        return decoder_output_variables
