from evaluation.seq2seq.beam_search import *
from utils.data_prep import *


def evaluate(config, test_articles, vocabulary, encoder, decoder, max_length):
    for i in range(len(test_articles)):
        # with_categories = config['train']['with_categories']
        # if with_categories:
        #     _, input_sentence = split_category_and_article(articles[i])
        # else:
        #     input_sentence = articles[i]

        input_sentence = test_articles[i].unked_article_tokens
        full_input_sentence = test_articles[i].article_tokens
        target_sentence = test_articles[i].abstract_tokens
        extended_vocab = test_articles[i].unknown_tokens

        # TODO: Go from indexes to words before printing full_input_sentence and target_sentence

        print('>', full_input_sentence, flush=True)
        print('=', target_sentence, flush=True)
        output_beams = evaluate_beams(config, vocabulary, encoder, decoder, input_sentence, full_input_sentence,
                                      max_length, extended_vocab)
        for beam in output_beams:
            output_words = beam.decoded_word_sequence
            output_sentence = ' '.join(output_words)
            print('<', str(beam.get_avg_score()), output_sentence, flush=True)
        print('', flush=True)


def evaluate_beams(config, vocabulary, encoder, decoder, input_variable, full_input_variable, max_length,
                   extended_vocab):
    # input_variable = indexes_from_sentence(vocabulary, sentence)
    input_variable = pad_seq(input_variable, max_length)
    input_length = max_length
    input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
    input_variable = input_variable.cuda() if use_cuda else input_variable

    full_input_variable = pad_seq(full_input_variable, max_length)
    full_input_variable = Variable(torch.LongTensor(full_input_variable)).unsqueeze(1)
    full_input_variable = full_input_variable.cuda() if use_cuda else full_input_variable

    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)

    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    # num_layers = config['model']['n_layers']
    # encoder_hidden = encoder_hidden.repeat(num_layers, 1, 1)

    expansions = config['evaluate']['expansions']
    keep_beams = config['evaluate']['keep_beams']
    return_beams = config['evaluate']['return_beams']

    # first decoder beam. input_hidden = encoder_hidden
    init_attention_weights = torch.zeros(max_length, max_length)
    beams = [Beam([], [], init_attention_weights, [], SOS_token, encoder_hidden, full_input_variable, extended_vocab)]
    for i in range(max_length):
        beams = expand_and_prune_beams(vocabulary, beams, encoder_outputs, decoder, expansions, keep_beams)

    pruned_beams = prune_beams(beams, return_beams)
    for beam in pruned_beams:
        beam.cut_attention_weights_at_sequence_length()
    return pruned_beams


# Calculate loss on the evaluation set. Does not modify anything.
def calculate_loss_on_eval_set(config, vocabulary, encoder, decoder, criterion, writer, epoch, max_length,
                               eval_articles, eval_titles):
    loss = 0
    for i in range(0, len(eval_articles)):
        with_categories = config['train']['with_categories']
        if with_categories:
            _, article = split_category_and_article(eval_articles[i])
        else:
            article = eval_articles[i]
        title = eval_titles[i]
        input_variable = indexes_from_sentence(vocabulary, article)
        input_variable = pad_seq(input_variable, max_length)
        input_length = max_length
        input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
        input_variable = input_variable.cuda() if use_cuda else input_variable

        target_variable = indexes_from_sentence(vocabulary, title)
        target_variable = Variable(torch.LongTensor(target_variable)).unsqueeze(1)
        target_variable = target_variable.cuda() if use_cuda else target_variable

        loss += calculate_loss_on_single_eval_article(config, encoder, decoder, criterion, input_variable,
                                                      target_variable, input_length)
    loss_avg = loss / len(eval_articles)
    writer.add_scalar('Evaluation loss', loss_avg, epoch)
    print("Evaluation set loss for epoch %d: %.4f" % (epoch, loss_avg), flush=True)


def calculate_loss_on_single_eval_article(config, encoder, decoder, criterion, input_variable, target_variable,
                                          input_length):
    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)

    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    # num_layers = config['model']['n_layers']
    # encoder_hidden = encoder_hidden.repeat(num_layers, 1, 1)

    decoder_input = Variable(torch.LongTensor([SOS_token]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    for di in range(target_variable.size()[0]):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, 1)
        topv, topi = decoder_output.data.topk(1)
        ni = topi  # next input, batch of top softmax scores
        decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
        loss += criterion(decoder_output, target_variable[di])

    return loss.data[0]
