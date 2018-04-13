import random

from torch.distributions import Categorical

from utils.data_prep import *
from utils.logger import *
import time


class Generator:
    def __init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 policy_criterion, batch_size, use_cuda, beta, num_monte_carlo_samples, sample_rate, negative_reward):
        self.vocabulary = vocabulary
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.mle_criterion = mle_criterion
        self.policy_criterion = policy_criterion
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.beta = beta
        self.num_monte_carlo_samples = num_monte_carlo_samples
        self.updates = 0
        self.cumulative_reward = 0.0
        self.sample_rate = sample_rate
        self.allow_negative_rewards = negative_reward
        self.use_trigram_check = True

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
        baseline = self.get_argmax_baseline(encoder_hidden, encoder_outputs, max_target_length,
                                            full_input_variable_batch, discriminator, full_target_variable_batch_2,
                                            extended_vocabs)

        # POLICY ITERATION START
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        full_policy_values = []
        accumulated_sequence = None

        policy_iteration_time_start = time.time()
        policy_iteration_break_early = False

        start_check_for_pad_and_eos = int(max_target_length / 3) * 2

        # Do policy iteration
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            m = Categorical(decoder_output)
            action = m.sample()
            log_prob = m.log_prob(action)
            full_policy_values.append(log_prob)

            # Assuming we can just do this now
            ni = action
            for token_index in range(0, len(ni)):
                if ni[token_index].data[0] >= self.vocabulary.n_words:
                    ni[token_index].data[0] = UNK_token
            decoder_input = ni.unsqueeze(1)

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

        policy_loss = 0
        reward = discriminator.evaluate(accumulated_sequence, full_target_variable_batch_2, extended_vocabs)
        adjusted_reward = reward - baseline

        print_log_sum = 0
        for i in range(0, len(full_policy_values)):
            print_log_sum += torch.sum(full_policy_values[i])
            loss = -full_policy_values[i] * reward
            policy_loss += torch.sum(loss) / self.batch_size
        print_log_sum = print_log_sum / self.batch_size

        timings[timings_var_policy_iteration] += (time.time() - policy_iteration_time_start)

        backprop_time_start = time.time()

        policy_loss.backward()

        clip = 2
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        timings[timings_var_backprop] += (time.time() - backprop_time_start)

        return print_log_sum.data[0], policy_loss.data[0], reward.mean(), baseline.mean(), adjusted_reward.mean()

    # TODO: Move to data_thingy util or something
    # using indexes
    def has_trigram(self, current_sequence, next_word):
        if len(current_sequence) < 3:
            return False
        word1 = current_sequence[-2]
        word2 = current_sequence[-1]
        word3 = next_word
        for i in range(2, len(current_sequence)):
            if current_sequence[i - 2] != word1:
                continue
            if current_sequence[i - 1] != word2:
                continue
            if current_sequence[i] != word3:
                continue
            # all equal = overlap
            return True
        return False

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
                # TODO: If topk > 5 the program should crash. Figure out if this is a case at all..
                # Using topk > 1 to compensate for possible trigram overlaps
                topv, topi = decoder_output.data.topk(10)
                next_words = []
                for batch in range(0, self.batch_size):
                    topk = 0
                    while 1:
                        next_word = topi[batch][topk]
                        if di > 2 and self.has_trigram(accumulated_sequence[batch].data, next_word):
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
