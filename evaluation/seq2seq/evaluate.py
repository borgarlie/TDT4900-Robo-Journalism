from evaluation.seq2seq.beam_search import *
from utils.data_prep import *


def evaluate(config, articles, titles, vocabulary, encoder, decoder, max_length):
    for i in range(len(articles)):
        _, input_sentence = split_category_and_article(articles[i])
        target_sentence = titles[i]
        print('>', input_sentence, flush=True)
        print('=', target_sentence, flush=True)
        output_beams = evaluate_beams(config, vocabulary, encoder, decoder, input_sentence, max_length)
        for beam in output_beams:
            output_words = beam.decoded_word_sequence
            output_sentence = ' '.join(output_words)
            print('<', str(beam.get_avg_score()), output_sentence, flush=True)
        print('', flush=True)


def evaluate_beams(config, vocabulary, encoder, decoder, sentence, max_length):
    input_variable = indexes_from_sentence(vocabulary, sentence)
    input_variable = pad_seq(input_variable, max_length)
    input_length = max_length
    input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
    input_variable = input_variable.cuda() if use_cuda else input_variable

    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)

    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    num_layers = config['model']['n_layers']
    encoder_hidden = encoder_hidden.repeat(num_layers, 1, 1)

    expansions = config['evaluate']['expansions']
    keep_beams = config['evaluate']['keep_beams']
    return_beams = config['evaluate']['return_beams']

    # first decoder beam. input_hidden = encoder_hidden
    init_attention_weights = torch.zeros(max_length, max_length)
    beams = [Beam([], [], init_attention_weights, [], SOS_token, encoder_hidden)]
    for i in range(max_length):
        beams = expand_and_prune_beams(vocabulary, beams, encoder_outputs, decoder, expansions, keep_beams)

    pruned_beams = prune_beams(beams, return_beams)
    for beam in pruned_beams:
        beam.cut_attention_weights_at_sequence_length()
    return pruned_beams
