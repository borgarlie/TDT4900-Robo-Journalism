import random

from torch import nn

from evaluation.seq2seq.evaluate import *
from utils.batching import *
from utils.data_prep import *
from utils.time_utils import *
from utils.logger import *


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

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                        full_input_variable, batch_size)
            log_output = torch.log(decoder_output.clamp(min=1e-8))
            loss += criterion(log_output, full_target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                        full_input_variable, batch_size)
            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax scores
            # if we produce an OOV word, then we need to replace input with UNK
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
            log_output = torch.log(decoder_output.clamp(min=1e-8))
            loss += criterion(log_output, full_target_variable[di])

    loss.backward()

    clip = 2
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

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

    log_message("Starting training")
    for epoch in range(start_epoch, n_epochs + 1):
        log_message("Starting epoch: %d" % epoch)
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
                lowest_loss = log_training_message(progress, itr, itr / n_iters * 100, print_loss_avg, lowest_loss)

        # log to tensorboard
        writer.add_scalar('loss', batch_loss_avg / num_batches, epoch)

        # save each epoch after epoch 7 with different naming
        if epoch > 7:
            log_message("Saving model")
            itr = epoch * num_batches
            _, total_runtime = time_since(start, itr / n_iters, total_runtime)
            save_state({
                'epoch': epoch,
                'runtime': total_runtime,
                'model_state_encoder': encoder.state_dict(),
                'model_state_decoder': decoder.state_dict(),
                'optimizer_state_encoder': encoder_optimizer.state_dict(),
                'optimizer_state_decoder': decoder_optimizer.state_dict()
            }, config['experiment_path'] + "/epoch%d_" % epoch + config['save']['save_file'])

        encoder.eval()
        decoder.eval()
        calculate_loss_on_eval_set(config, vocabulary, encoder, decoder, criterion, writer, epoch, max_article_length,
                                   eval_pairs)
        # run beam search evaluation for a smaller subset
        if len(eval_pairs) > 20:
            eval_pairs_subset = eval_pairs[0:20]
            evaluate(config, eval_pairs_subset, vocabulary, encoder, decoder, max_length=max_article_length)
        encoder.train()
        decoder.train()


def save_state(state, filename):
    torch.save(state, filename)
