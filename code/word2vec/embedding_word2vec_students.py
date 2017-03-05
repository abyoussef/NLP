from random import randint
import numpy as np
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

from gensim.models import word2vec


def avg_word2vec(model, dataset='data/snli.test'):
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


def cosine_similarity(a, b):
    assert len(a) == len(b), 'vectors need to have the same size'
    cos_sim = -1
    assert cos_sim >= 0, "TODO (assignment): You need to implement cosine_similarity"
    #########
    # TO DO : IMPLEMENT THE COSINE SIMILARITY BETWEEN a AND b
    #########
    return cos_sim


def most_similar(idx, array_embeddings, array_sentences):
    query_sentence = array_sentences[idx]
    query_embed = array_embeddings[idx]
    list_scores = {}
    for i in range(idx) + range(idx + 1, len(array_sentences)):
        list_scores[i] = cosine_similarity(query_embed, array_embeddings[i])
    closest_idx = max(list_scores, key=list_scores.get)
    #########
    # TO DO : output the 5 most similar sentences
    #########
    
    print 'The query :\n'
    print query_sentence + '\n'
    print 'is most similar to\n'
    print array_sentences[closest_idx]
    print 'with a score of : {0}'.format(list_scores[closest_idx])

    return closest_idx

def most_5_similar(idx, array_embeddings, array_sentences):
    query_sentence = array_sentences[idx]
    query_embed = array_embeddings[idx]
    list_scores = {}
    for i in range(idx) + range(idx + 1, len(array_sentences)):
        list_scores[i] = cosine_similarity(query_embed, array_embeddings[i])
    #########
    # TO DO : find and output the 5 most similar sentences
    #########
    closest_5_idx = []

    assert len(closest_5_idx) == 5, "TODO (assignment): You need to implement most_5_similar function"

    return closest_5_idx


def IDF(dataset='data/snli.test'):
    # Compute IDF (Inverse Document Frequency). Here a "document" is a sentence.
    # word2idf['peach'] = IDF(peach)
    word2idf = {}
    with open(dataset) as f:
        for line in f:
            sentence = line.split()
            for word in sentence:
                break
    assert len(word2idf)>0, "The IDF function has not been implemented yet"
    return word2idf

def avg_word2vec_idf(model, word2idf, dataset='data/snli.test'):
    # TODO : Modify this to have a weighted (idf weights) average of the word vectors
    array_sentences = []
    array_embeddings = []
    with open(dataset) as f:
        for line in f:
            avgword2vec = None
            for word in line.split():
                # get embedding (if it exists) of each word in the sentence
                if word in model.vocab:
                    if avgword2vec is None:
                        # TODO : ADD WEIGHTS
                        avgword2vec = model[word]
                    else:
                        # TODO : ADD WEIGHTS
                        avgword2vec = avgword2vec + model[word]
            # if at least one word in the sentence has a word embeddings :
            if avgword2vec is not None:
                # TODO : NORMALIZE BY THE SUM OF THE WEIGHTS
                avgword2vec = avgword2vec / len(avgword2vec)  # normalize sum
                array_sentences.append(line)
                array_embeddings.append(avgword2vec)
    print 'Generated embeddings for {0} sentences from {1} dataset.'.format(len(array_sentences), dataset)
    return array_sentences, array_embeddings

if __name__ == "__main__":

    if False: # FIRST PART
        sentences = word2vec.Text8Corpus('data/text8')

        # Train a word2vec model
        embedding_size = 200
        your_model = word2vec.Word2Vec(sentences, size=embedding_size)
        #########
        # TO DO : Report from INFO :
            # - total number of raw words found in the corpus.
            # - number of words retained in the vocabulary (with min_count = 5)
        #########

        # Train a word2vec model with phrases
        # bigram_transformer = gensim.models.Phrases(sentences)
        # your_model_phrase = Word2Vec(bigram_transformer[sentences], size=200)

    if True: # SECOND PART

        """
        Investigating word2vec word embeddings space
        """
        # Loading model trained on words
        model = word2vec.Word2Vec.load('models/text8.model')

        # Loading model enhanced with phrases (2-grams)
        model_phrase = word2vec.Word2Vec.load('models/text8.phrase.model')

        # Words that are similar are close in the sense of the cosine similarity.
        sim = model.similarity('woman', 'man')
        print 'Printing word similarity between "woman" and "man" : {0}'.format(sim)

        # And words that appear in the same context have similar word embeddings.
        model.most_similar(['paris'])
        model_phrase.most_similar(['paris'])

        # Compositionality and structure in word2vec space
        model.most_similar(positive=['woman', 'king'], negative=['man'])

        #########
        # TO DO : Compute similarity (france, berlin, germany)
        #########


    if False: # THIRD PART
        """
        Sentence embeddings with average(word2vec)
        """
        data_path = 'data/snli.test'
        array_sentences, array_embeddings = avg_word2vec(model, dataset=data_path)

        #########
        # TO DO : do the TODOs in cosine_similarity
        #########
        query_idx =  777 # random sentence
        assert query_idx < len(array_sentences) # little check

        # For the next line to work, you need to implement the "cosine_similarity" function.
        # array_sentences[closest_idx] will be the closest sentence to array_sentences[query_idx].
        closest_idx = most_similar(query_idx, array_embeddings, array_sentences)

        #########
        # TO DO : Implement the most_5_similar function to output the 5 sentences that are closest to the query.
        # TO DO : Report the 5 most similar sentences to query_idx = 777
        #########
        closest_5_idx = most_5_similar(query_idx, array_embeddings, array_sentences)

        for idx in closest_5_idx:
            break
            #########
            # TO DO: Print the 5 most similar sentences to query_idx using closest_5_idx, array_sentences, array_embeddings
            #########

    if False: # FOURTH PART
        #######
        # Weighted average of word vectors with IDF.
        #######
        word2idf = IDF(data_path)
        array_sentences_idf, array_embeddings_idf = avg_word2vec_idf(model, word2idf, dataset=data_path)
        closest_idx_idf = most_similar(query_idx, array_embeddings_idf, array_sentences_idf)
