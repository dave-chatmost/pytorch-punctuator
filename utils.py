import errno
import os


def load_vocab(vocab_path, extra_word_list=[], encoding='utf8'):
    n = len(extra_word_list)
    with open(vocab_path, encoding=encoding) as vocab_file:
        vocab = { word.strip(): i + n for i, word in enumerate(vocab_file) }
    for i, word in enumerate(extra_word_list):
            vocab[word] = i
    return vocab

def num_param(model):
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params

def mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    vocab = load_vocab(sys.argv[1], ["<UNK>", "<END>"])
    print(vocab)
    vocab = load_vocab(sys.argv[1])
    print(vocab)
