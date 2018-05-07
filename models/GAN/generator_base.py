from torch.distributions import Categorical

from utils.data_prep import *
from utils.logger import *
import time


class GeneratorBase:
    def __init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 batch_size, use_cuda, beta, num_monte_carlo_samples, sample_rate, negative_reward, use_trigram_check,
                 use_running_avg_baseline, discriminator_batch_size):
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
        self.cumulative_reward = 0
        self.sample_rate = sample_rate
        self.allow_negative_rewards = negative_reward
        self.use_trigram_check = use_trigram_check
        self.use_running_avg_baseline = use_running_avg_baseline
        self.discriminator_batch_size = discriminator_batch_size
        self.rollout_batchsize = self.batch_size * self.num_monte_carlo_samples

        self.MASK = torch.LongTensor([UNK_token] * self.batch_size).cuda()
        self.MONTE_CARLO_MASK = torch.LongTensor([UNK_token] * self.rollout_batchsize).cuda()
        self.CREATE_FAKE_MASK = torch.LongTensor([UNK_token] * self.discriminator_batch_size).cuda()
        self.UPPER_BOUND = torch.LongTensor([self.vocabulary.n_words] * self.batch_size).cuda()
        self.MONTE_CARLO_UPPER_BOUND = torch.LongTensor([self.vocabulary.n_words] * self.rollout_batchsize).cuda()
        self.CREATE_FAKE_UPPER_BOUND \
            = torch.LongTensor([self.vocabulary.n_words] * self.discriminator_batch_size).cuda()

        self.EOS_MATRIX = torch.LongTensor([EOS_token] * self.batch_size).cuda()
        self.PAD_MATRIX = torch.LongTensor([PAD_token] * self.batch_size).cuda()
        self.EOS_MATRIX_MONTE_CARLO = torch.LongTensor([EOS_token] * self.rollout_batchsize).cuda()
        self.PAD_MATRIX_MONTE_CARLO = torch.LongTensor([PAD_token] * self.rollout_batchsize).cuda()
        self.EOS_MATRIX_CREATE_FAKE = torch.LongTensor([EOS_token] * self.discriminator_batch_size).cuda()
        self.PAD_MATRIX_CREATE_FAKE = torch.LongTensor([PAD_token] * self.discriminator_batch_size).cuda()

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

            if accumulated_sequence is None:
                accumulated_sequence = Variable(ni).clone()
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, Variable(ni)), 1)

            # Remove UNK before setting next input to decoder
            unk_check_time_start = time.time()
            ni = ni.squeeze(1)
            ni = where(ni < self.UPPER_BOUND, ni, self.MASK)
            decoder_input = Variable(ni.unsqueeze(1))
            timings[timings_var_unk_check] += time.time() - unk_check_time_start

        baseline = discriminator.evaluate(accumulated_sequence, full_target_variable_batch_2, extended_vocabs)
        return baseline

    # Used to create fake data samples to train the discriminator
    # Returned values as batched sentences as variables
    def create_samples(self, input_variable_batch, full_input_variable_batch, input_lengths, max_sample_length,
                       pad_length, discriminator_batch_size, sample=False):

        self.encoder.eval()
        self.decoder.eval()

        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity
        decoder_input = Variable(torch.LongTensor([SOS_token] * discriminator_batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        accumulated_sequence = None

        if max_sample_length == pad_length:
            max_sample_length = max_sample_length - 1

        ni = None

        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_sample_length):

            create_fake_inner_time_start = time.time()
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               discriminator_batch_size)
            timings[timings_var_create_fake_inner] += (time.time() - create_fake_inner_time_start)

            if sample:
                m = Categorical(decoder_output)
                ni = m.sample()
                # Remove UNK before setting next input to decoder
                unk_check_time_start = time.time()
                ni = ni.data
                ni = where(ni < self.CREATE_FAKE_UPPER_BOUND, ni, self.CREATE_FAKE_MASK)
                decoder_input = Variable(ni.unsqueeze(1))
                timings[timings_var_unk_check] += time.time() - unk_check_time_start
            else:
                topv, topi = decoder_output.data.topk(1)
                ni = topi  # next input, batch of top softmax scores
                # Remove UNK before setting next input to decoder
                unk_check_time_start = time.time()
                ni = ni.squeeze(1)
                ni = where(ni < self.CREATE_FAKE_UPPER_BOUND, ni, self.CREATE_FAKE_MASK)
                decoder_input = Variable(ni.unsqueeze(1))
                timings[timings_var_unk_check] += time.time() - unk_check_time_start

            if accumulated_sequence is None:
                accumulated_sequence = decoder_input
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, decoder_input), 1)

        # Adding EOS for those that are max_length and PAD to the rest
        last_tokens = where(ni > self.EOS_MATRIX_CREATE_FAKE, self.EOS_MATRIX_CREATE_FAKE,
                            self.PAD_MATRIX_CREATE_FAKE)
        accumulated_sequence = torch.cat((accumulated_sequence, Variable(last_tokens.unsqueeze(1))), 1)

        decoder_outputs = accumulated_sequence.data.cpu().numpy()
        decoder_outputs_padded = [pad_seq(s.tolist(), pad_length) for s in decoder_outputs]
        decoder_output_variables = Variable(torch.LongTensor(decoder_outputs_padded))
        if self.use_cuda:
            decoder_output_variables = decoder_output_variables.cuda()

        self.encoder.train()
        self.decoder.train()

        return decoder_output_variables
