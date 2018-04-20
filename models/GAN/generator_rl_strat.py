from torch.distributions import Categorical

from models.GAN.generator_base import GeneratorBase
from utils.data_prep import *
from utils.logger import *
import time


class GeneratorRlStrat(GeneratorBase):
    def __init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 batch_size, use_cuda, beta, num_monte_carlo_samples, sample_rate, negative_reward, use_trigram_check,
                 use_running_avg_baseline):
        GeneratorBase.__init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 batch_size, use_cuda, beta, num_monte_carlo_samples, sample_rate, negative_reward, use_trigram_check,
                 use_running_avg_baseline)

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
        accumulated_sequence = None

        policy_iteration_time_start = time.time()
        policy_iteration_break_early = False

        start_check_for_pad_and_eos = int(max_target_length / 3) * 2
        num_samples = 0

        # Policy iteration
        for di in range(max_target_length):
            num_samples += 1

            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            m = Categorical(decoder_output)
            action = m.sample()
            log_prob = m.log_prob(action)
            full_policy_values.append(log_prob)

            cloned_action = action.unsqueeze(1).clone()
            # Update accumulated sequence
            if accumulated_sequence is None:
                accumulated_sequence = cloned_action
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, cloned_action), 1)

            ni = action
            for token_index in range(0, len(ni)):
                if ni[token_index].data[0] >= self.vocabulary.n_words:
                    ni[token_index].data[0] = UNK_token
            decoder_input = ni.unsqueeze(1)

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

        if self.use_running_avg_baseline:
            self.cumulative_reward += reward.mean().data[0]
            self.updates += 1
            # Calculate running average baseline
            avg = self.cumulative_reward / self.updates
            baseline = Variable(torch.cuda.FloatTensor([avg]))
            print_baseline = baseline.data[0]

        adjusted_reward = reward - baseline
        if not self.allow_negative_rewards:
            for j in range(0, len(adjusted_reward.data)):
                if adjusted_reward.data[j] < 0.0:
                    adjusted_reward.data[j] = 0.0

        # TODO: Alternative: Test with instead of setting negative values to 0,
        # instead just scale them by divinding by 100 or something?

        print_log_sum = 0
        for i in range(0, len(full_policy_values)):
            print_log_sum += torch.sum(full_policy_values[i])
            loss = -full_policy_values[i] * adjusted_reward
            policy_loss += torch.sum(loss) / self.batch_size
        print_log_sum = print_log_sum / self.batch_size

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
                   reward.mean(), print_baseline, adjusted_reward.mean()
        else:
            return total_loss.data[0], total_loss.data[0], policy_loss.data[0], print_log_sum.data[0], \
                   reward.mean(), print_baseline, adjusted_reward.mean()
