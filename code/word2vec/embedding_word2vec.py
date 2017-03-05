from random import randint
import numpy as np
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')


from gensim.models import word2vec


def most_similar(idx, array_embeddings, array_sentences):
    query_sentence = array_sentences[idx]
    query_embed = array_embeddings[idx]
    list_scores = {}
    for i in range(idx) + range(idx + 1, len(array_sentences)):
        list_scores[i] = cosine_similarity(query_embed, array_embeddings[i])
    argmax_score = max(list_scores, key=list_scores.get)

    # MODIFY THIS : output the 5 most similar sentences

    print 'The query :\n'
    print query_sentence + '\n'
    print 'is most similar to\n'
    print array_sentences[argmax_score]
    print 'with a score of : {0}'.format(list_scores[argmax_score])

    return argmax_score


def cosine_similarity(a, b):
    assert len(a) == len(b), 'vectors need to have the same size'
    cos_sim = 0
    for k in range(len(a)):
        cos_sim += a[k] * b[k]
    cos_sim = cos_sim / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim


def avg_word2vec(dataset='word2vec/data/snli.test'):
    array_sentences = []
    array_embeddings = []
    with open(dataset) as f:
        for line in f:
            avgword2vec = None
            for word in line.split():
                # get embedding (if it exists) of each word in the sentence
                if word in model.vocab:
                    if avgword2vec is None:
                        avgword2vec = model[word]
                    else:
                        avgword2vec = avgword2vec + model[word]
            # if at least one word in the sentence has a word embeddings :
            if avgword2vec is not None:
                avgword2vec = avgword2vec / len(avgword2vec)  # normalize sum
                array_sentences.append(line)
                array_embeddings.append(avgword2vec)
    print 'Generated embeddings for {0} sentences from {1} dataset.'.format(len(array_sentences), dataset)
    return array_sentences, array_embeddings


if __name__ == "__main__":

    if True:
        sentences = word2vec.Text8Corpus('data/text8')

        # Train a word2vec model
        embedding_size = 200
        your_model = word2vec.Word2Vec(sentences, size=embedding_size)

        # Train a word2vec model with phrases
        # bigram_transformer = gensim.models.Phrases(sentences)
        # your_model_phrase = Word2Vec(bigram_transformer[sentences], size=200)

    if False: # Change this to True if you want to run it

        """
        Investigating word2vec word embeddings space
        """
        # Loading model trained on words
        model = word2vec.Word2Vec.load('models/text8.model')
        # Loading model enhanced with phrases (2-grams)
        model_phrase = word2vec.Word2Vec.load('models/text8.phrase.model')

        # Words that are similar are close in the sense of the cosine similarity.
        print model.similarity('woman', 'man')

        # And words that appear in the same context have similar word embeddings.
        model.most_similar(['california'])
        model_phrase.most_similar(['california'])

        # Compositionality and structure in word2vec space
        model.most_similar(positive=['woman', 'king'], negative=['man'])

        # REMOVE THIS : QUESTION
        model.most_similar(positive=['france', 'berlin'], negative=['germany'])

    if False:
        """
        Sentence embeddings with average(word2vec)
        """
        data_path = 'data/snli.test'
        array_sentences, array_embeddings = avg_word2vec(dataset=data_path)

        query_idx =  1723 # random sentence
        assert query_idx < len(array_sentences) # little check
        most_similar(query_idx, array_embeddings, array_sentences)