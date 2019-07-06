# nltk.download()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from glove import Corpus, Glove
import argparse
import codecs
import sys
import logging
import configparser
import numpy as np
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# lines = ['Hello this is a tutorial on how to convert the word in an integer format', 'this is a beautiful day', 'Jack is going to office']

def remove_stopwords(lines):
    stop_words = set(stopwords.words('english'))
    lines_without_stopwords = []  # stop words contain the set of stop words
    for line in lines:
        temp_line = []
        for word in line.split():
            if word not in stop_words:
                temp_line.append(word)
        lines_without_stopwords.append(' '.join(temp_line))
    return lines_without_stopwords


def lemmatize(lines):
    wordnet_lemmatizer = WordNetLemmatizer()
    lines_with_lemmas = []  # stop words contain the set of stop words
    for line in lines:
        temp_line = []
        for word in line.split():
            temp_line.append(wordnet_lemmatizer.lemmatize(word))
        lines_with_lemmas.append(' '.join(temp_line))
    return lines_with_lemmas


def preprocessing(lines):
    lines = remove_stopwords(lines)
    lines = lemmatize(lines)
    new_lines = []
    for line in lines:
        new_lines.append(line.split())  # new lines has the new format
    return new_lines


def read_lines(f_name):
    fin = codecs.open(f_name, 'r', encoding='utf-8', errors='ignore')
    return fin.readlines()


def load_glove_bin(f_name):
    model = glove.load(f_name)
    return model


def load_glove_vec(f_name):
    vocab = list()
    vecs = list()
    word2vec = dict()
    with open(f_name, 'r') as f:
        header = f.readline()
        if len(header.split()) == 2:
            vocab_size, vector_size = map(int, header.split())
        elif len(header.split()) > 2:
            parts = header.rstrip().split(" ")
            word, vec = parts[0].strip(), list(map(np.float32, parts[1:]))
            vocab.append(word)
            vecs.append(vec)
            word2vec[word] = vec
        for _, line in enumerate(f):
            parts = line.rstrip().split(" ")
            word, vec = parts[0].strip(), list(map(np.float32, parts[1:]))
            vocab.append(word)
            vecs.append(vec)
            word2vec[word] = vec
    return word2vec, vocab, vecs


def save_txt(glove_bin_model, f_out):
    fout = codecs.open(f_out, 'w', encoding='utf-8')
    for k, v in glove_bin_model.dictionary.items():
        fout.write(str(k) + ' ' + ' '.join(map(str, glove.word_vectors[v].tolist())) + '\n')


def get_embedding(glove_bin_model, word):
    return glove_bin_model.word_vectors[glove_bin_model.dictionary[word]]


def read_params(config):
    params = dict()
    params['epochs'] = int(config['epochs'])
    params['no_components'] = int(config['no_components'])
    params['lr'] = float(config['lr'])
    params['max_count'] = int(config['max_count'])
    params['window'] = int(config['window'])
    params['no_threads'] = int(config['no_threads'])
    params['verbose'] = config['verbose']
    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str, default=None, help='path to corpus')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--no_components', type=int, default=300, help='number of latent dimensions')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate for SGD estimation.')
    parser.add_argument('--max_count', type=int, default=5, help='parameters for the weighting function')
    parser.add_argument('--window', type=int, default=10, help='the length of the (symmetric) context window used for cooccurrence.')
    parser.add_argument('--no_threads', type=int, default=8, help='number of training threads')
    parser.add_argument('--verbose', type=bool, default=True, help='print progress messages if True')
    logger.info('parsing argument...\n')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read('config.cfg')
    params = read_params(config['glove'])

    logger.info('reading data...\n')
    lines = read_lines(args.corpus_path)
    logger.info('preprocessing: remove stopwords and lemma...\n')
    lines = preprocessing(lines)

    corpus = Corpus()
    # training the corpus to generate the co occurence matrix which is used in GloVe
    logger.info('generating co occurence matrix...\n')
    corpus.fit(lines, params['window'])
    # creating a Glove object which will use the matrix created in the above lines to create embeddings
    glove = Glove(no_components=params['no_components'], learning_rate=params['lr'], max_count=params['max_count'])
    logger.info('training...\n')
    glove.fit(corpus.matrix, epochs=params['epochs'], no_threads=params['no_threads'], verbose=params['verbose'])
    glove.add_dictionary(corpus.dictionary)
    glove.save('movie_review_embedding.model')
    save_txt(glove, 'movie_review_embedding.txt')

    # print('------ *EXAMPLE* ------')
    # print(glove.word_vectors[glove.dictionary['tutorial']])

    # model = glove.load('movie_review_embedding.model')
    # print(model.word_vectors[model.dictionary['tutorial']])
