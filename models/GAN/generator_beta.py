from utils.data_prep import *
from utils.logger import *
import time


class GeneratorBeta:
    def __init__(self, vocabulary, encoder, decoder, batch_size, use_cuda):
        self.vocabulary = vocabulary
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        # intermediate variables used when sampling multiple times with different length
        self.forced_decoder_hidden = None
        self.forced_encoder_outputs = None

        for param in self.encoder.parameters():
            param.requires_grad = False

        for param in self.decoder.parameters():
            param.requires_grad = False

    def generate_sequence(self, full_input_variable_batch, max_sample_length, previous_token, sampled_token,
                          initial_sequence):

        monte_carlo_encoder_time_start = time.time()
        start_check_for_pad_and_eos = int(max_sample_length / 3) * 2

        # run previous state
        _, decoder_hidden, _ = self.decoder(previous_token, self.forced_decoder_hidden, self.forced_encoder_outputs,
                                            full_input_variable_batch, self.batch_size)
        self.forced_decoder_hidden = decoder_hidden

        if initial_sequence is not None:
            start = len(initial_sequence.data[0]) + 1
            decoder_output_variables = torch.cat((initial_sequence, sampled_token), 1)
        else:
            start = 1
            decoder_output_variables = sampled_token

        timings[timings_var_monte_carlo_encoder] += (time.time() - monte_carlo_encoder_time_start)

        decoder_input = sampled_token

        monte_carlo_sampling_break_early = False
        for di in range(start, max_sample_length):
            monte_carlo_inner_time_start = time.time()
            decoder_output, decoder_hidden, _ \
                = self.decoder(decoder_input, decoder_hidden, self.forced_encoder_outputs, full_input_variable_batch,
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

    # resets and calculate new values for encoder outputs and decoder hidden state
    def calculate_initial_encoder_decoder_states(self, input_variable_batch, input_lengths):
        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        self.forced_encoder_outputs = encoder_outputs
        self.forced_decoder_hidden = encoder_hidden

    def update_params(self, external_encoder, external_decoder):
        self.encoder.load_state_dict(external_encoder.state_dict())
        self.decoder.load_state_dict(external_decoder.state_dict())

        if self.use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()
