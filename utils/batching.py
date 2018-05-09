from utils.data_prep import *


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def prepare_batch(batch_size, summary_pairs, max_article_length, max_abstract_length):
    input_seqs = []
    full_input_seqs = []
    target_seqs = []
    full_target_seqs = []
    extended_vocabs = []
    full_target_seqs_2 = []

    for i in range(batch_size):
        input_seqs.append(summary_pairs[i].unked_article_tokens)
        full_input_seqs.append(summary_pairs[i].article_tokens)
        target_seqs.append(summary_pairs[i].unked_abstract_tokens)
        full_target_seqs.append(summary_pairs[i].abstract_tokens)
        extended_vocabs.append(summary_pairs[i].unknown_tokens)
        full_target_seqs_2.append(summary_pairs[i].unked_abstract_tokens)

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, full_input_seqs, target_seqs, full_target_seqs, extended_vocabs,
                           full_target_seqs_2), key=lambda p: len(p[0]), reverse=True)
    input_seqs, full_input_seqs, target_seqs, full_target_seqs, extended_vocabs, full_target_seqs_2 = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [max_article_length for s in input_seqs]
    input_padded = [pad_seq(s, max_article_length) for s in input_seqs]
    full_input_padded = [pad_seq(s, max_article_length) for s in full_input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max_abstract_length) for s in target_seqs]
    full_target_padded = [pad_seq(s, max_abstract_length) for s in full_target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    full_input_var = Variable(torch.LongTensor(full_input_padded))  # No need to transpose full input
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)
    full_target_var = Variable(torch.LongTensor(full_target_padded)).transpose(0, 1)

    if use_cuda:
        input_var = input_var.cuda()
        full_input_var = full_input_var.cuda()
        target_var = target_var.cuda()
        full_target_var = full_target_var.cuda()

    return input_var, full_input_var, input_lengths, target_var, full_target_var, target_lengths, extended_vocabs, full_target_seqs_2
