import sys
import os

from torch.distributions import Categorical

sys.path.append('../..')  # ugly dirtyfix for imports to work

from models.seq2seq.decoder import PointerGeneratorDecoder
from models.seq2seq.encoder import EncoderRNN
from preprocess.preprocess_pointer import *
from utils.data_prep import *


minimum_abstract_length = 4


def generate_argmax_summaries(vocabulary, encoder, decoder, summary_pairs, max_article_length, max_abstract_length):
    fake_data = []
    print_every = 500
    for i in range(0, len(summary_pairs)):
        article = summary_pairs[i].unked_article_tokens
        # TODO: Is it necessary to PAD here?
        input_variable = pad_seq(article, max_article_length)
        input_length = max_article_length
        input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
        input_variable = input_variable.cuda() if use_cuda else input_variable

        full_article = summary_pairs[i].article_tokens
        full_input_variable = pad_seq(full_article, max_article_length)
        full_input_variable = Variable(torch.LongTensor(full_input_variable)).unsqueeze(0)
        full_input_variable = full_input_variable.cuda() if use_cuda else full_input_variable

        # fake_sample = generate_single_argmax_summary(vocabulary, encoder, decoder, input_variable, full_input_variable,
        #                                              input_length, max_abstract_length)

        fake_sample = generate_single_sampled_summary(vocabulary, encoder, decoder, input_variable, full_input_variable,
                                                     input_length, max_abstract_length)

        fake_data.append(' '.join(fake_sample))
        if (i+1) % print_every == 0:
            print("Done creating : %d / %d samples" % ((i+1), len(summary_pairs)), flush=True)
    print("Done creating fake data", flush=True)
    return fake_data


def generate_single_argmax_summary(vocabulary, encoder, decoder, input_variable, full_input_variable, input_length,
                                   max_abstract_length):
    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)
    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    decoder_input = Variable(torch.LongTensor([SOS_token]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden
    decoded_word_sequence = []
    for di in range(max_abstract_length):
        decoder_output, decoder_hidden, decoder_attention \
            = decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable, 1)
        topv, topi = decoder_output.data.topk(3)
        next_word = topi[0][0]
        # need special logic to ensure minimum words
        if next_word == EOS_token or next_word == PAD_token:
            if di >= minimum_abstract_length:
                break
            counter = 1
            while next_word == EOS_token or next_word == PAD_token:
                next_word = topi[0][counter]
                counter += 1

        # replace OOV words with UNK
        if next_word >= vocabulary.n_words:
            next_word = UNK_token

        decoded_word = vocabulary.index2word[next_word]
        decoded_word_sequence.append(decoded_word)
        # next input to decoder
        ni = [next_word]  # next input, batch of top softmax scores
        decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
    return decoded_word_sequence


def generate_single_sampled_summary(vocabulary, encoder, decoder, input_variable, full_input_variable, input_length,
                                   max_abstract_length):
    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)
    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    decoder_input = Variable(torch.LongTensor([SOS_token]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden
    decoded_word_sequence = []
    for di in range(max_abstract_length):
        decoder_output, decoder_hidden, decoder_attention \
            = decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable, 1)

        m = Categorical(decoder_output)
        action = m.sample()
        next_word = action.data[0]

        # need special logic to ensure minimum words
        if next_word == EOS_token or next_word == PAD_token:
            if di >= minimum_abstract_length:
                break
            while next_word == EOS_token or next_word == PAD_token:
                action = m.sample()
                next_word = action.data[0]

        # replace OOV words with UNK
        if next_word >= vocabulary.n_words:
            next_word = UNK_token

        decoded_word = vocabulary.index2word[next_word]
        decoded_word_sequence.append(decoded_word)
        # next input to decoder
        ni = [next_word]  # next input, batch of top softmax scores
        decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
    return decoded_word_sequence


def load_state(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return state['model_state_encoder'], state['model_state_decoder']
    else:
        raise FileNotFoundError


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        if len(sys.argv) < 2:
            print("Expected 1 argument: [0] = GPU (0 or 1)", flush=True)
            exit()
        torch.cuda.set_device(int(sys.argv[1]))
        print("Using GPU: %s" % sys.argv[1], flush=True)

    relative_path = "../../data/cnn_pickled/cnn_pointer_50k"
    # relative_path = "../../data/ntb_pickled/ntb_pointer_30k"
    hidden_size = 128
    embedding_size = 100
    n_layers = 1
    dropout_p = 0.0
    load_file = "../../models/pretrained_models/cnn/epoch13_cnn_test1.pth.tar"
    # load_file = "../../models/pretrained_models/ntb/ntb_pretrain_2epochs.tar"

    summary_pairs, vocabulary = load_dataset(relative_path)

    encoder = EncoderRNN(vocabulary.n_words, embedding_size, hidden_size, n_layers=n_layers)

    max_article_length = max(len(pair.article_tokens) for pair in summary_pairs) + 1
    max_abstract_length = max(len(pair.abstract_tokens) for pair in summary_pairs) + 1

    decoder = PointerGeneratorDecoder(hidden_size, embedding_size, vocabulary.n_words, max_length=max_article_length,
                                      n_layers=n_layers, dropout_p=dropout_p)

    try:
        model_state_encoder, model_state_decoder = load_state(load_file)
        encoder.load_state_dict(model_state_encoder)
        decoder.load_state_dict(model_state_decoder)
    except FileNotFoundError as e:
        print("No file found: exiting", flush=True)
        exit()

    encoder.eval()
    decoder.eval()

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    # summary_pairs = summary_pairs[0:1000]

    # generate data
    fake_data = generate_argmax_summaries(vocabulary, encoder, decoder, summary_pairs, max_article_length,
                                          max_abstract_length)

    # for sample in fake_data:
    #     print(sample, flush=True)

    print("Writing to file", flush=True)

    fake_data_save_file = "../../data/cnn_fake_data/cnn_13epoch_sampled.abstract.txt"
    with open(fake_data_save_file, 'w') as file:
        for sample in fake_data:
            file.write(sample)
            file.write("\n")

    print("Done", flush=True)
