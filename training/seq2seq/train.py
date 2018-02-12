import random

from torch import nn

from evaluation.seq2seq.evaluate import *
from utils.batching import *
from utils.data_prep import *
from utils.time_utils import *


# Train one batch
def train(config, vocabulary, input_variable, full_input_variable, input_lengths, target_variable, full_target_variable,
          target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    batch_size = config['train']['batch_size']
    teacher_forcing_ratio = config['train']['teacher_forcing_ratio']

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    max_target_length = max(target_lengths)
    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, None)

    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    num_layers = config['model']['n_layers']
    encoder_hidden = encoder_hidden.repeat(num_layers, 1, 1)

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # TODO: FIX all below here.

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                        full_input_variable, batch_size)
            # TODO: is log correct here?
            loss += criterion(torch.log(decoder_output), full_target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                        full_input_variable, batch_size)
            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax scores

            # if we produce an OOV word, then we need to replace input with UNK
            for token_index in range(0, len(ni)):  # TODO: Is it >= or > ?
                if ni[token_index] >= vocabulary.n_words:
                    ni[token_index] = UNK_token

            decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))

            # TODO: is log correct here?
            loss += criterion(torch.log(decoder_output), full_target_variable[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]


def train_iters(config, training_pairs, eval_pairs, vocabulary, encoder, decoder, max_article_length,
                max_abstract_length, encoder_optimizer, decoder_optimizer, writer, start_epoch=1, total_runtime=0):

    start = time.time()
    print_loss_total = 0  # Reset every print_every
    lowest_loss = 999

    n_epochs = config['train']['num_epochs']
    batch_size = config['train']['batch_size']
    print_every = config['log']['print_every']

    criterion = nn.NLLLoss()

    num_batches = int(len(training_pairs) / batch_size)
    n_iters = num_batches * n_epochs

    print("Starting training", flush=True)
    for epoch in range(start_epoch, n_epochs + 1):
        print("Starting epoch: %d" % epoch, flush=True)
        batch_loss_avg = 0

        # shuffle articles and titles (equally)
        # c = list(zip(articles, titles))
        random.shuffle(training_pairs)  # TODO: Check that this is ok
        # articles_shuffled, titles_shuffled = zip(*c)

        # split into batches
        training_pair_batches = list(chunks(training_pairs, batch_size))
        # title_batches = list(chunks(titles_shuffled, batch_size))

        for batch in range(num_batches):
            input_variable, full_input_variable, input_lengths, target_variable, full_target_variable, target_lengths \
                = prepare_batch(batch_size, training_pair_batches[batch], max_article_length, max_abstract_length)

            loss = train(config, vocabulary, input_variable, full_input_variable, input_lengths, target_variable,
                         full_target_variable, target_lengths,
                         encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss
            batch_loss_avg += loss
            # calculate number of batches processed
            itr = (epoch-1) * num_batches + batch + 1

            if itr % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                progress, total_runtime = time_since(start, itr / n_iters, total_runtime)
                start = time.time()
                print('%s (%d %d%%) %.4f' % (progress, itr, itr / n_iters * 100, print_loss_avg), flush=True)
                if print_loss_avg < lowest_loss:
                    lowest_loss = print_loss_avg
                    print(" ^ Lowest loss so far", flush=True)

        # log to tensorboard
        writer.add_scalar('loss', batch_loss_avg / num_batches, epoch)

        # save each epoch
        print("Saving model", flush=True)
        itr = epoch * num_batches
        _, total_runtime = time_since(start, itr / n_iters, total_runtime)
        save_state({
            'epoch': epoch,
            'runtime': total_runtime,
            'model_state_encoder': encoder.state_dict(),
            'model_state_decoder': decoder.state_dict(),
            'optimizer_state_encoder': encoder_optimizer.state_dict(),
            'optimizer_state_decoder': decoder_optimizer.state_dict()
        }, config['experiment_path'] + "/" + config['save']['save_file'])

        # TODO: Fix eval
        # encoder.eval()
        # decoder.eval()
        # calculate_loss_on_eval_set(config, vocabulary, encoder, decoder, criterion, writer, epoch, max_article_length,
        #                            eval_articles, eval_titles)
        # encoder.train()
        # decoder.train()


def save_state(state, filename):
    torch.save(state, filename)
