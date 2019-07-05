import nltk
# nltk.download()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lines = ['Hello this is a tutorial on how to convert the word in an integer format', 'this is a beautiful day', 'Jack is going to office']

def remove_stopwords(lines):
    stop_words = set(stopwords.words('english'))
    print('stopwords: ', stop_words)
    print('before remove stopwords: ', lines)
    lines_without_stopwords = []  # stop words contain the set of stop words
    for line in lines:
        temp_line = []
        for word in line.split():
            if word not in stop_words:
                temp_line.append(word)
        lines_without_stopwords.append(' '.join(temp_line))
    print('after remove stopwords: ',lines_without_stopwords)
    return lines_without_stopwords


def lemmatize(lines):
    print('before remove lemma: ', lines)
    wordnet_lemmatizer = WordNetLemmatizer()
    lines_with_lemmas = []  # stop words contain the set of stop words
    for line in lines:
        temp_line = []
        for word in line.split():
            temp_line.append(wordnet_lemmatizer.lemmatize(word))
        lines_with_lemmas.append(' '.join(temp_line))
    print('after remove lemma: ', lines)
    return lines_with_lemmas


def preprocessing(lines):
    print('--------------------')
    lines = remove_stopwords(lines)
    print('--------------------')
    lines = lemmatize(lines)
    print('--------------------')
    new_lines = []
    for line in lines:
        new_lines.append(line.split())  # new lines has the new format lines=new_lines
    print(new_lines)
    print('--------------------')
    return new_lines


def read_data(f_name):
    fin = open(f_name, 'r')
    return fin.readlines()


from glove import Corpus, Glove
corpus = Corpus()
# training the corpus to generate the co occurence matrix which is used in GloVe
lines = preprocessing(lines)
corpus.fit(lines, window=10)
# # creating a Glove object which will use the matrix created in the above lines to create embeddings
# # We can set the learning rate as it uses Gradient Descent and number of components
glove = Glove(no_components=5, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove_embedding.model')

print(glove.word_vectors[glove.dictionary['tutorial']])