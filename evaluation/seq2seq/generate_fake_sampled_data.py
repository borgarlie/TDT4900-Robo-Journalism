import sys
import os

from torch.distributions import Categorical

sys.path.append('../..')  # ugly dirtyfix for imports to work

from models.seq2seq.decoder import PointerGeneratorDecoder
from models.seq2seq.encoder import EncoderRNN
from preprocess.preprocess_pointer import *
from utils.batching import *


def load_state(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return state['model_state_encoder'], state['model_state_decoder']
    else:
        raise FileNotFoundError


class SampleDataGenerator:
    def __init__(self, encoder, decoder, vocabulary, use_cuda):
        self.encoder = encoder
        self.decoder = decoder
        self.vocabulary = vocabulary
        self.use_cuda = use_cuda
        self.print_every = 100

    def initialise_and_create_samples(self, summary_pairs, max_sample_length, pad_length, batch_size, sample=True):

        batches = list(chunks(summary_pairs, batch_size))
        generated_samples = []

        for i in range(0, len(batches)):
            if (i+1) % self.print_every == 0:
                print("Creating samples for batch: %d / %d" % (i+1, len(batches)), flush=True)

            input_variable, full_input_variable, input_lengths, _, _, _, _, _ \
                = prepare_batch(batch_size, batches[i], max_article_length, pad_length)

            batched_generated_samples = self.create_samples(input_variable, full_input_variable, input_lengths,
                                                            max_sample_length, pad_length, batch_size, sample)

            # convert samples to text
            for j in range(0, len(batched_generated_samples)):
                generated_sample = get_sentence_from_tokens_unked(batched_generated_samples[j], self.vocabulary)
                generated_samples.append(generated_sample)

        return generated_samples

    # Used to create fake data samples to train the discriminator
    # Returned values as batched sentences as variables
    def create_samples(self, input_variable_batch, full_input_variable_batch, input_lengths, max_sample_length,
                       pad_length, discriminator_batch_size, sample=True):

        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity
        decoder_input = Variable(torch.LongTensor([SOS_token] * discriminator_batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        decoder_outputs = [[] for _ in range(0, discriminator_batch_size)]
        create_fake_sample_break_early = False

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
                for token_index in range(0, len(ni)):
                    if ni[token_index].data[0] >= self.vocabulary.n_words:
                        ni[token_index].data[0] = UNK_token
                decoder_input = ni.unsqueeze(1)
                ni = ni.unsqueeze(1).data
            else:
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
        return decoder_outputs_padded


if __name__ == '__main__':

    # setup GPU
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if len(sys.argv) < 2:
            print("Expected 1 argument: [0] = GPU (0 or 1)", flush=True)
            exit()
        device_number = int(sys.argv[1])
        if device_number > -1:
            torch.cuda.set_device(device_number)
            print("Using GPU: %s" % sys.argv[1], flush=True)
        else:
            print("Not setting specific GPU", flush=True)

    # load dataset and vocabulary
    print("Loading dataset and vocabulary", flush=True)
    relative_path = "../../data/cnn_pickled/cnn_pointer_50k"
    summary_pairs, vocabulary = load_dataset(relative_path)

    max_article_length = max(len(pair.article_tokens) for pair in summary_pairs) + 1
    max_abstract_length = max(len(pair.abstract_tokens) for pair in summary_pairs) + 1

    # setup the models
    hidden_size = 128
    embedding_size = 100
    n_layers = 1
    dropout_p = 0.0

    encoder = EncoderRNN(vocabulary.n_words, embedding_size, hidden_size, n_layers=n_layers)
    decoder = PointerGeneratorDecoder(hidden_size, embedding_size, vocabulary.n_words, max_length=max_article_length,
                                      n_layers=n_layers, dropout_p=dropout_p)

    # load saved models
    print("Loading models", flush=True)
    load_file = "../../models/pretrained_models/cnn/epoch13_cnn_test1.pth.tar"
    try:
        model_state_encoder, model_state_decoder = load_state(load_file)
        encoder.load_state_dict(model_state_encoder)
        decoder.load_state_dict(model_state_decoder)
    except FileNotFoundError as e:
        print("No file found: exiting", flush=True)
        exit()

    # set eval and cuda
    encoder.eval()
    decoder.eval()

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # setup generator
    generator = SampleDataGenerator(encoder, decoder, vocabulary, use_cuda)

    # sampling params
    max_sample_length = max_abstract_length
    pad_length = max_abstract_length
    batch_size = 64
    sample = True
    validation_pairs = 13000

    total_articles = len(summary_pairs)
    generate_articles_length = total_articles - validation_pairs
    # Append remainder to validation set so that the generation set has exactly a multiple of batch size
    validation_pairs += generate_articles_length % batch_size
    generate_articles_length = total_articles - validation_pairs

    sample_pairs = summary_pairs[:generate_articles_length]

    # run sampling
    generate_samples = True
    if generate_samples:
        print("Generating data for %d samples" % len(sample_pairs), flush=True)
        generated_samples = generator.initialise_and_create_samples(sample_pairs, max_sample_length, pad_length,
                                                                    batch_size, sample)
        # write to file
        print("Writing to file", flush=True)
        sampled_num = 1
        sampled_data_save_file = "../../data/cnn_sampled_data/generated_samples_%d.abstract.txt" % sampled_num
        with open(sampled_data_save_file, 'w') as file:
            for sample in generated_samples:
                file.write(sample)
                file.write("\n")

    # generate validation data
    generate_validation_data = True
    batch_size = 32
    if generate_validation_data:
        validation_pairs = int(validation_pairs / batch_size) * batch_size
        validation_pairs = summary_pairs[-validation_pairs:]
        print("Generating validation data for %d samples" % len(validation_pairs), flush=True)
        generated_validation_samples = generator.initialise_and_create_samples(validation_pairs, max_sample_length,
                                                                               pad_length, batch_size, sample)
        # write validation data to file
        validation_sampled_num = 1
        validation_sampled_data_save_file \
            = "../../data/cnn_validation_sampled_data/generated_validation_samples_%d.abstract.txt" \
              % validation_sampled_num
        with open(validation_sampled_data_save_file, 'w') as file:
            for sample in generated_validation_samples:
                file.write(sample)
                file.write("\n")

    print("Done creating samples", flush=True)
