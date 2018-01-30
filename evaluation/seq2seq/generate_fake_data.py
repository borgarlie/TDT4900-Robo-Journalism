import sys
import os

sys.path.append('../..')  # ugly dirtyfix for imports to work

from models.seq2seq.decoder import AttnDecoderRNN
from models.seq2seq.encoder import EncoderRNN
from preprocess import preprocess
from utils.data_prep import *


def generate_argmax_summaries(vocabulary, encoder, decoder, articles, max_length):
    fake_data = []
    print_every = 100
    for i in range(0, len(articles)):
        _, article = split_category_and_article(articles[i])
        input_variable = indexes_from_sentence(vocabulary, article)
        input_variable = pad_seq(input_variable, max_length)
        input_length = max_length
        input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
        input_variable = input_variable.cuda() if use_cuda else input_variable
        fake_sample = generate_single_argmax_summary(vocabulary, encoder, decoder, input_variable, input_length)
        fake_data.append(' '.join(fake_sample))
        if (i+1) % print_every == 0:
            print("Done creating : %d / %d samples" % ((i+1), len(articles)), flush=True)
    print("Done creating fake data", flush=True)
    return fake_data


def generate_single_argmax_summary(vocabulary, encoder, decoder, input_variable, input_length):
    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)
    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    decoder_input = Variable(torch.LongTensor([SOS_token]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden
    decoded_word_sequence = []
    for di in range(input_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, 1)
        topv, topi = decoder_output.data.topk(1)
        ni = topi  # next input, batch of top softmax scores
        decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
        next_word = topi[0][0]
        if next_word == EOS_token or next_word == PAD_token:
            break
        decoded_word = vocabulary.index2word[next_word]
        decoded_word_sequence.append(decoded_word)
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

    relative_path = "../../data/ntb_preprocessed/ntb_80_5cat.unk"
    hidden_size = 128
    n_layers = 1
    dropout_p = 0.1
    load_file = "../../models/pretrained_models/pretrained_seq2seq.pth.tar"

    articles, titles, vocabulary = preprocess.generate_vocabulary(relative_path, -1, True)

    encoder = EncoderRNN(vocabulary.n_words, hidden_size, n_layers=n_layers)

    max_length = max(len(article.split(">>>")[1].strip().split(' ')) for article in articles) + 1

    decoder = AttnDecoderRNN(hidden_size, vocabulary.n_words, max_length=max_length, n_layers=n_layers,
                             dropout_p=dropout_p)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    try:
        model_state_encoder, model_state_decoder = load_state(load_file)
        encoder.load_state_dict(model_state_encoder)
        decoder.load_state_dict(model_state_decoder)
    except FileNotFoundError as e:
        print("No file found: exiting", flush=True)
        exit()

    encoder.eval()
    decoder.eval()

    articles = articles[0:5000]

    # generate data
    fake_data = generate_argmax_summaries(vocabulary, encoder, decoder, articles, max_length)

    # for sample in fake_data:
    #     print(sample, flush=True)

    print("Writing to file", flush=True)

    fake_data_save_file = "../../data/ntb_fake_data/test_generated_fake_data.unk.title.txt"
    with open(fake_data_save_file, 'w') as file:
        for sample in fake_data:
            file.write(sample)
            file.write("\n")

    print("Done", flush=True)
