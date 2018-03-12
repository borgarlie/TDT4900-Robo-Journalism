from evaluation.seq2seq.beam_search import *
from utils.data_prep import *
from utils.logger import *


def evaluate(config, test_articles, vocabulary, encoder, decoder, max_length):
    for i in range(len(test_articles)):
        input_sentence = test_articles[i].unked_article_tokens
        full_input_sentence = test_articles[i].article_tokens
        full_target_sentence = test_articles[i].abstract_tokens
        extended_vocab = test_articles[i].unknown_tokens

        full_input_sentence_unpacked = get_sentence_from_tokens(full_input_sentence, vocabulary, extended_vocab)
        full_target_sentence_unpacked = get_sentence_from_tokens(full_target_sentence, vocabulary, extended_vocab)

        log_message('> %s' % full_input_sentence_unpacked)
        log_message('= %s' % full_target_sentence_unpacked)
        output_beams = evaluate_beams(config, vocabulary, encoder, decoder, input_sentence, full_input_sentence,
                                      max_length, extended_vocab)
        for beam in output_beams:
            output_words = beam.decoded_word_sequence
            output_sentence = ' '.join(output_words)
            log_message('< %s %s' % (str(beam.get_avg_score()), output_sentence))
            log_message('')


def evaluate_argmax(vocabulary, test_articles, encoder, decoder, max_length):
    for i in range(len(test_articles)):
        input_sentence = test_articles[i].unked_article_tokens
        full_input_sentence = test_articles[i].article_tokens
        full_target_sentence = test_articles[i].abstract_tokens
        extended_vocab = test_articles[i].unknown_tokens
        input_length = [len(input_sentence)]
        input_variable = Variable(torch.LongTensor(input_sentence)).unsqueeze(1)
        input_variable = input_variable.cuda() if use_cuda else input_variable
        full_input_variable = Variable(torch.LongTensor(full_input_sentence)).unsqueeze(0)
        full_input_variable = full_input_variable.cuda() if use_cuda else full_input_variable
        decoder_outputs = evaluate_single_argmax(vocabulary, input_variable, full_input_variable, input_length,
                                                 max_length, encoder, decoder)
        full_target_sentence_unpacked = get_sentence_from_tokens(full_target_sentence, vocabulary, extended_vocab)
        full_generated_sentence_unpacked = get_sentence_from_tokens(decoder_outputs, vocabulary, extended_vocab)
        print("TARGET SENTENCE >>> " + full_target_sentence_unpacked, flush=True)
        print("GENERATED SENTENCE >>> " + full_generated_sentence_unpacked, flush=True)


def evaluate_single_argmax(vocabulary, input_variable, full_input_variable, input_lengths, max_sample_length, encoder,
                           decoder):

    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, None)
    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)

    decoder_input = Variable(torch.LongTensor([SOS_token]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    decoder_outputs = []
    # Without teacher forcing: use its own predictions as the next input
    for di in range(max_sample_length):
        decoder_output, decoder_hidden, decoder_attention \
            = decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable, 1)

        topv, topi = decoder_output.data.topk(1)
        ni = topi  # next input, batch of top softmax scores

        decoder_outputs.append(ni[0][0])

        if ni[0][0] == EOS_token or ni[0][0] == PAD_token:
            break

        if ni[0][0] >= vocabulary.n_words:
            ni[0][0] = UNK_token
        decoder_input = Variable(ni)

    return decoder_outputs


def evaluate_beams(config, vocabulary, encoder, decoder, input_variable, full_input_variable, max_length,
                   extended_vocab):
    input_length = len(input_variable)
    input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
    input_variable = input_variable.cuda() if use_cuda else input_variable
    full_input_variable = Variable(torch.LongTensor(full_input_variable)).unsqueeze(0)
    full_input_variable = full_input_variable.cuda() if use_cuda else full_input_variable

    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)

    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    # num_layers = config['model']['n_layers']
    # encoder_hidden = encoder_hidden.repeat(num_layers, 1, 1)

    expansions = config['evaluate']['expansions']
    keep_beams = config['evaluate']['keep_beams']
    return_beams = config['evaluate']['return_beams']

    # first decoder beam. input_hidden = encoder_hidden
    init_attention_weights = torch.zeros(input_length, input_length)
    beams = [Beam([], [], init_attention_weights, [], SOS_token, encoder_hidden, full_input_variable, extended_vocab)]
    for i in range(input_length):
        beams = expand_and_prune_beams(vocabulary, beams, encoder_outputs, decoder, expansions, keep_beams)

    pruned_beams = prune_beams(beams, return_beams)
    for beam in pruned_beams:
        beam.cut_attention_weights_at_sequence_length()

    return pruned_beams


# Calculate loss on the evaluation set. Does not modify anything.
def calculate_loss_on_eval_set(config, vocabulary, encoder, decoder, criterion, writer, epoch, max_length,
                               eval_pairs, use_logger=True):
    loss = 0
    for i in range(0, len(eval_pairs)):
        article = eval_pairs[i].unked_article_tokens
        full_article = eval_pairs[i].article_tokens
        abstract = eval_pairs[i].abstract_tokens

        # TODO: Not sure if we need pad here. But it might make a difference regarding loss?
        input_variable = pad_seq(article, max_length)
        input_length = max_length
        input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
        input_variable = input_variable.cuda() if use_cuda else input_variable

        full_article_variable = pad_seq(full_article, max_length)
        full_article_variable = Variable(torch.LongTensor(full_article_variable)).unsqueeze(0)
        full_article_variable = full_article_variable.cuda() if use_cuda else full_article_variable

        full_target_variable = Variable(torch.LongTensor(abstract)).unsqueeze(1)
        full_target_variable = full_target_variable.cuda() if use_cuda else full_target_variable

        loss += calculate_loss_on_single_eval_article(config, vocabulary, encoder, decoder, criterion, input_variable,
                                                      full_target_variable, input_length, full_article_variable)
    loss_avg = loss / len(eval_pairs)
    writer.add_scalar('Evaluation_loss', loss_avg, epoch)
    message = "Evaluation set loss for epoch %d: %.4f" % (epoch, loss_avg)
    if use_logger:
        log_message(message)
    else:
        print(message, flush=True)


def calculate_loss_on_single_eval_article(config, vocabulary, encoder, decoder, criterion, input_variable,
                                          full_target_variable, input_length, full_article_variable):
    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)

    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    # num_layers = config['model']['n_layers']
    # encoder_hidden = encoder_hidden.repeat(num_layers, 1, 1)

    decoder_input = Variable(torch.LongTensor([SOS_token]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    for di in range(full_target_variable.size()[0]):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                    full_article_variable, 1)
        topv, topi = decoder_output.data.topk(1)
        ni = topi  # next input, batch of top softmax scores

        for token_index in range(0, len(ni)):
            if ni[token_index][0] >= vocabulary.n_words:
                ni[token_index][0] = UNK_token
        decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
        log_output = torch.log(decoder_output.clamp(min=1e-8))
        loss += criterion(log_output, full_target_variable[di])

    return loss.data[0]
