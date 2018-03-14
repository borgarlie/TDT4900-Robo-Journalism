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

    def generate_sequence(self, input_variable_batch, full_input_variable_batch, input_lengths, max_sample_length,
                          initial_sequence):

        monte_carlo_encoder_time_start = time.time()

        decoder_input, decoder_hidden, encoder_outputs \
            = self.get_decoder_hidden_state(input_variable_batch, input_lengths, initial_sequence)

        decoder_output_variables = initial_sequence
        timings[timings_var_monte_carlo_encoder] += (time.time() - monte_carlo_encoder_time_start)

        updated = False
        monte_carlo_sampling_break_early = False
        for di in range(len(initial_sequence), max_sample_length):
            monte_carlo_inner_time_start = time.time()
            decoder_output, decoder_hidden, _ \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            timings[timings_var_monte_carlo_inner] += (time.time() - monte_carlo_inner_time_start)

            monte_carlo_inner_transpose_time_start = time.time()

            ni = decoder_output.data.multinomial(1)
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)
            ni_transposed = ni.transpose(0, 1)
            decoder_input_batch = Variable(ni_transposed)

            timings[timings_var_monte_carlo_inner_transpose] += (time.time() - monte_carlo_inner_transpose_time_start)

            monte_carlo_cat_time_start = time.time()

            decoder_output_variables = torch.cat((decoder_output_variables, decoder_input_batch), 0)

            timings[timings_var_monte_carlo_cat] += (time.time() - monte_carlo_cat_time_start)

            if not updated:
                self.forced_decoder_hidden = decoder_hidden
                updated = True

            if is_whole_batch_pad_or_eos(ni):
                monte_carlo_sampling[decode_breaking_monte_carlo_sampling] += di
                monte_carlo_sampling[monte_carlo_sampling_num] += 1
                monte_carlo_sampling_break_early = True
                break

        if not monte_carlo_sampling_break_early:
            monte_carlo_sampling[decode_breaking_monte_carlo_sampling] += max_sample_length - 1
            monte_carlo_sampling[monte_carlo_sampling_num] += 1

        return decoder_output_variables

    def get_decoder_hidden_state(self, input_variable_batch, input_lengths, initial_sequence):
        if len(initial_sequence) == 1:
            self.calculate_initial_encoder_decoder_states(input_variable_batch, input_lengths)
        last_generated_batch = initial_sequence[-1]
        last_generated_batch = last_generated_batch.unsqueeze(1)
        return last_generated_batch, self.forced_decoder_hidden, self.forced_encoder_outputs

    # resets and calculate new values for encoder outputs and decoder hidden state
    def calculate_initial_encoder_decoder_states(self, input_variable_batch, input_lengths):
        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity
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
