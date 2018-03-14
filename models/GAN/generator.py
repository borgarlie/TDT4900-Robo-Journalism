from utils.data_prep import *
from utils.logger import *
import time


class Generator:
    def __init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion, policy_criterion,
                 batch_size, use_cuda, beta, generator_beta, num_monte_carlo_samples):
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
        self.generator_beta = generator_beta
        self.num_monte_carlo_samples = num_monte_carlo_samples

    # discriminator is used to calculate reward
    # target batch is used for MLE
    def train_on_batch(self, input_variable_batch, full_input_variable_batch, input_lengths, full_target_variable_batch,
                       target_lengths, discriminator, max_monte_carlo_length):

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        max_target_length = max(target_lengths)

        init_encoder_time_start = time.time()

        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)

        timings[timings_var_init_encoder] += (time.time() - init_encoder_time_start)

        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity

        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        # Testing with setting a lower upper limit to monte carlo sequence length
        monte_carlo_length = min(max_target_length, max_monte_carlo_length)

        mle_loss = 0
        policy_loss = 0
        total_reward = 0
        adjusted_reward = 0
        accumulated_sequence = None
        accumulated_sequence_argmax = None
        full_sequence_rewards = []
        full_policy_values = []

        baseline_time_start = time.time()
        baseline_break_early = False

        # calculate baseline value
        for di in range(max_target_length):
            baseline_inner_time_start = time.time()
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            timings[timings_var_baseline_inner] += (time.time() - baseline_inner_time_start)

            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token

            decoder_input = Variable(ni)
            if di == 0:
                accumulated_sequence_argmax = decoder_input
            else:
                accumulated_sequence_argmax = torch.cat((accumulated_sequence_argmax, decoder_input), 1)

            if is_whole_batch_pad_or_eos(ni):
                decode_breakings[decode_breaking_baseline] += di
                baseline_break_early = True
                break

        if not baseline_break_early:
            decode_breakings[decode_breaking_baseline] += max_target_length - 1

        baseline = discriminator.evaluate(accumulated_sequence_argmax)

        timings[timings_var_baseline] += (time.time() - baseline_time_start)

        # Printing mean baseline value
        # print(baseline.mean().data[0], flush=True)

        # reset values
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        policy_iteration_time_start = time.time()
        policy_iteration_break_early = False
        # Do policy iteration
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)

            log_output = torch.log(decoder_output.clamp(min=1e-8))
            mle_loss += self.mle_criterion(log_output, full_target_variable_batch[di])

            next_input = decoder_output.data.multinomial(1)
            for token_index in range(0, len(next_input)):
                if next_input[token_index][0] >= self.vocabulary.n_words:
                    next_input[token_index][0] = UNK_token

            decoder_input = Variable(next_input)
            decoder_input_transposed = decoder_input.transpose(1, 0)
            if di == 0:
                # TODO: Do we need to transpose this? dunno
                accumulated_sequence = decoder_input_transposed
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, decoder_input_transposed), 0)

            # calculate policy value
            policy_target = Variable(next_input.squeeze(1))
            current_policy_value = self.policy_criterion(log_output, policy_target)
            full_policy_values.append(current_policy_value)
            # calculate policy loss using monte carlo search
            accumulated_reward = 0

            monte_carlo_time_start = time.time()

            for _ in range(self.num_monte_carlo_samples):
                sample = self.generator_beta.generate_sequence(input_variable_batch, full_input_variable_batch,
                                                               input_lengths, monte_carlo_length, accumulated_sequence)

                monte_carlo_outer_time_start = time.time()
                sample = sample.transpose(1, 0)
                accumulated_reward += discriminator.evaluate(sample)
                timings[timings_var_monte_carlo_outer] += (time.time() - monte_carlo_outer_time_start)

            timings[timings_var_monte_carlo] += (time.time() - monte_carlo_time_start)

            reward = accumulated_reward / self.num_monte_carlo_samples
            full_sequence_rewards.append(reward)
            total_reward += reward  # used for printing only

            # Break the policy iteration loop if all the variables in the batch is at EOS or PAD
            if is_whole_batch_pad_or_eos(next_input):
                decode_breakings[decode_breaking_policy] += di
                policy_iteration_break_early = True
                break

        if not policy_iteration_break_early:
            decode_breakings[decode_breaking_policy] += max_target_length - 1

        for i in range(0, len(full_sequence_rewards)):
            # Not allowing negative rewards
            temp_adjusted_reward = full_sequence_rewards[i] - baseline
            for j in range(0, len(temp_adjusted_reward.data)):
                if temp_adjusted_reward.data[j] < 0.0:
                    temp_adjusted_reward.data[j] = 0.0
            adjusted_reward += temp_adjusted_reward
            current_policy_loss = temp_adjusted_reward * full_policy_values[i]
            reduced_policy_loss = current_policy_loss.mean()
            policy_loss += reduced_policy_loss

        # Test to look at difference between argmax and multinomial samples
        # argmax_1 = accumulated_sequence_argmax[0].data
        # print(get_sentence_from_tokens_unked(argmax_1, self.vocabulary), flush=True)
        # multinomial_1 = accumulated_sequence.transpose(1, 0)[0].data
        # print(get_sentence_from_tokens_unked(multinomial_1, self.vocabulary), flush=True)
        # exit()

        total_loss = self.beta * policy_loss + (1 - self.beta) * mle_loss

        timings[timings_var_policy_iteration] += (time.time() - policy_iteration_time_start)

        backprop_time_start = time.time()

        total_loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        timings[timings_var_backprop] += (time.time() - backprop_time_start)

        total_reward = (total_reward / max_target_length).mean()
        adjusted_reward = (adjusted_reward / max_target_length).mean()

        return total_loss.data[0], mle_loss.data[0], policy_loss.data[0], total_reward, adjusted_reward

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

    def update_generator_beta_params(self):
        self.generator_beta.update_params(self.encoder, self.decoder)
