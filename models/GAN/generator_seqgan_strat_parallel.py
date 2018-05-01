import random

from torch.distributions import Categorical

from models.GAN.generator_base import GeneratorBase
from utils.data_prep import *
from utils.logger import *
import time


class BaseSendObj:
    def __init__(self, tag):
        self.tag = tag


class NewAction(BaseSendObj):
    def __init__(self, tag, action, decoder_hidden, initial_sequence):
        BaseSendObj.__init__(self, tag)
        self.action = action.data.cpu().numpy()
        self.decoder_hidden = decoder_hidden.data.cpu().numpy()
        self.initial_sequence = initial_sequence.data.cpu().numpy()


class NewBatch(BaseSendObj):
    def __init__(self, tag, encoder_outputs, full_input_variable_batch, action, decoder_hidden):
        BaseSendObj.__init__(self, tag)
        self.encoder_outputs = encoder_outputs.data.cpu().numpy()
        self.full_input_variable_batch = full_input_variable_batch.data.cpu().numpy()
        self.action = action.data.cpu().numpy()
        self.decoder_hidden = decoder_hidden.data.cpu().numpy()


class GeneratorSeqGanStratParallel(GeneratorBase):
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
                       full_target_variable_batch_2, process_connections):

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

                # running with multiple processes
                # NOTE: As of now: len(process_connections) must be equal to num_monte_carlo_samples

                # Send next action and hidden states to the processes
                send_objects = []
                if di == 0:
                    # Send all the information
                    for rank in range(0, len(process_connections)):
                        send_obj = NewBatch("new_batch", encoder_outputs, full_input_variable_batch, action,
                                            decoder_hidden)
                        send_objects.append(send_obj)
                        process_connections[rank].send(send_obj)
                else:
                    # Only send action, hidden state and accumulated sequence
                    for rank in range(0, len(process_connections)):
                        send_obj = NewAction("new_action", action, decoder_hidden, accumulated_sequence)
                        send_objects.append(send_obj)
                        process_connections[rank].send(send_obj)

                # receive data from processes
                samples = []
                for rank in range(0, len(process_connections)):
                    returned_sample = process_connections[rank].recv()
                    samples.append(returned_sample)

                # get reward per sample
                temp_reward = 0
                for n in range(0, self.num_monte_carlo_samples):
                    sample = Variable(torch.LongTensor(samples[n])).cuda()
                    current_reward = discriminator.evaluate(sample, full_target_variable_batch_2, extended_vocabs)
                    temp_reward += current_reward

                # calculate average reward
                avg_reward = temp_reward / self.num_monte_carlo_samples
                full_sequence_rewards.append(avg_reward)

                # add cumulative reward to calculate running average baseline
                if self.use_running_avg_baseline:
                    self.cumulative_reward += avg_reward.mean().data[0]
                    self.updates += 1
                timings[timings_var_monte_carlo] += (time.time() - monte_carlo_time_start)

            # Get top1 for the next input to the decoder
            topv, topi = decoder_output.data.topk(1)
            ni = topi

            # Update accumulated sequence
            if accumulated_sequence is None:
                accumulated_sequence = Variable(ni).clone()
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, Variable(ni)), 1)

            # Remove UNK before setting next input to decoder
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)

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


# send_obj = NewBatch("new_batch", encoder_outputs, full_input_variable_batch, action, decoder_hidden)
# send_obj = NewAction("new_action", action, decoder_hidden, accumulated_sequence)

def p_monte_carlo_expansion(max_sample_length, vocabulary, decoder, batch_size, rank, master_conn, device):

    # setup process
    if device > -1:
        torch.cuda.set_device(device)

    # Temporary variables that we only receive once per batch
    encoder_outputs = None
    full_input_variable_batch = None

    # run process until receiving None
    while True:

        received_object = master_conn.recv()
        if received_object is None:
            break

        if received_object.tag == "new_batch":
            encoder_outputs = Variable(torch.FloatTensor(received_object.encoder_outputs)).cuda()
            full_input_variable_batch = Variable(torch.LongTensor(received_object.full_input_variable_batch)).cuda()
            action = Variable(torch.LongTensor(received_object.action)).cuda()
            decoder_hidden = Variable(torch.FloatTensor(received_object.decoder_hidden)).cuda()
            initial_sequence = None
        elif received_object.tag == "new_action":
            action = Variable(torch.LongTensor(received_object.action)).cuda()
            decoder_hidden = Variable(torch.FloatTensor(received_object.decoder_hidden)).cuda()
            initial_sequence = Variable(torch.LongTensor(received_object.initial_sequence)).cuda()

        start_check_for_pad_and_eos = int(max_sample_length / 3) * 2

        current_action = action.unsqueeze(1).clone()
        if initial_sequence is not None:
            start = len(initial_sequence.data[0]) + 1
            decoder_output_variables = torch.cat((initial_sequence, current_action), 1)
        else:
            start = 1
            decoder_output_variables = current_action

        # Prepare next decoder input with UNK
        for token_index in range(0, len(action)):
            if action[token_index].data[0] >= vocabulary.n_words:
                action[token_index].data[0] = UNK_token
        decoder_input = action.unsqueeze(1)

        for di in range(start, max_sample_length):
            decoder_output, decoder_hidden, _ \
                = decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch, batch_size)

            m = Categorical(decoder_output)
            action = m.sample()
            cloned_action = action.unsqueeze(1).clone()

            decoder_output_variables = torch.cat((decoder_output_variables, cloned_action), 1)

            for token_index in range(0, len(action)):
                if action[token_index].data[0] >= vocabulary.n_words:
                    action[token_index].data[0] = UNK_token
            decoder_input = action.unsqueeze(1)

            if di > start_check_for_pad_and_eos:
                if is_whole_batch_pad_or_eos(decoder_input.data):
                    break

        send_data = decoder_output_variables.data.cpu().numpy()

        master_conn.send(send_data)
