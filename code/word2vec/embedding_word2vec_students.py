from random import randint
import numpy as np
import logging
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
from gensim.models import word2vec
from gensim.models.tfidfmodel import df2idf # df2idf(docfreq, totaldocs, log_base=2.0, add=0.0)

def avg_word2vec(model, dataset='data/snli.test'):
    array_sentences = []
    array_embeddings = []
    with open(dataset) as f:
        for line in f:
            avgword2vec = None
            wordcount   = 0
            for word in line.split():
                # get embedding (if it exists) of each word in the sentence
                if word in model.wv.vocab:
                    if avgword2vec is None:
                        avgword2vec = model[word]
                    else:
                        avgword2vec = avgword2vec + model[word]
                    wordcount += 1
            # if at least one word in the sentence has a word embeddings :
            if avgword2vec is not None:
                avgword2vec = avgword2vec   / wordcount   #  / len(avgword2vec)  # normalize sum
                array_sentences.append(line)
                array_embeddings.append(avgword2vec)
    print 'Generated embeddings for {0} sentences from {1} dataset.'.format(len(array_sentences), dataset)
    return array_sentences, array_embeddings


def cosine_similarity(a, b):
    assert len(a) == len(b), 'vectors need to have the same size'
    #cos_sim = -1

    cos_sim = np.divide(np.inner(a,b) ,  np.linalg.norm(a) * np.linalg.norm(b) )
    #print cos_sim
    # assert cos_sim >= 0, "TODO (assignment): You need to implement cosine_similarity"
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
    # Sort array decreasingly
    sorted_list_scores = sorted(list_scores.items(), key=lambda kv:kv[1] , reverse= True)
    for i in range(5) :
        closest_5_idx.append(sorted_list_scores[i][0])
    assert len(closest_5_idx) == 5, "TODO (assignment): You need to implement most_5_similar function"

    return closest_5_idx


def IDF(dataset='data/snli.test'):
    # Compute IDF (Inverse Document Frequency). Here a "document" is a sentence.
    # word2idf['peach'] = IDF(peach)
    word2idf = {}
    # wordapp['peach'] = Number of appearance of 'peach' in sentences with counting only once for each sentence
    # even if multiple appearance in one sentence
    wordapp = {}
    numline = 0
    with open(dataset) as f:
        for line in f:
            sentence = line.split()
            unique_element_sentence = list(set(sentence))
            for word in unique_element_sentence:
                if word in wordapp.keys() :
                    wordapp[word] = wordapp[word] + 1
                else:
                    wordapp[word] = 1
            numline = numline + 1
    for word in wordapp.keys() :
        word2idf[word] =  df2idf(wordapp[word],numline)
    assert len(word2idf)>0, "The IDF function has not been implemented yet"
    return word2idf,wordapp,numline

def avg_word2vec_idf(model, word2idf, dataset='data/snli.test'):
    # TODO : Modify this to have a weighted (idf weights) average of the word vectors
    array_sentences = []
    array_embeddings = []
    normsum = 0
    with open(dataset) as f:
        for line in f:
            avgword2vec = None
            for word in line.split():
                # get embedding (if it exists) of each word in the sentence
                if word in model.wv.vocab:
                    if avgword2vec is None:
                        # TODO : ADD WEIGHTS
                        avgword2vec = model[word] * word2idf[word]
                    else:
                        # TODO : ADD WEIGHTS
                        avgword2vec = avgword2vec + model[word] * word2idf[word]
                    normsum = normsum + word2idf[word]
            # if at least one word in the sentence has a word embeddings :
            if avgword2vec is not None:
                # TODO : NORMALIZE BY THE SUM OF THE WEIGHTS
                avgword2vec = avgword2vec / normsum  # normalize sum
                array_sentences.append(line)
                array_embeddings.append(avgword2vec)
    print 'Generated embeddings for {0} sentences from {1} dataset.'.format(len(array_sentences), dataset)
    return array_sentences, array_embeddings

if __name__ == "__main__":

    if False: # FIRST PART
        sentences = word2vec.Text8Corpus('data/text8')
        print sentences
        # Train a word2vec model
        embedding_size = 200
        your_model = word2vec.Word2Vec(sentences, size=embedding_size, min_count=5)
        print your_model.syn0.shape
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
        print model.most_similar(['paris'])
        print model_phrase.most_similar(['paris'])

        # Compositionality and structure in word2vec space
        model.most_similar(positive=['woman', 'king'], negative=['man'])

        #########
        # TO DO : Q1) Compute similarity (apple, mac) (apple, peach) (banana, peach)
        #########
        print "Q1) "
        sim_1 = model.similarity('apple', 'mac')
        print 'Similarity between "apple" and "mac" : {0}'.format(sim_1)
        sim_2 = model.similarity('apple', 'peach')
        print 'Similarity between "apple" and "peach" : {0}'.format(sim_2)
        sim_3 = model.similarity('banana', 'peach')
        print 'Similarity between "banana" and "peach" : {0}'.format(sim_3)
        #########
        # TO DO : Q2) a Closest word to the word Difficult :
        #########
        print "Q2) a "
        print model.most_similar(['difficult'])
        print model_phrase.most_similar(['difficult'])
        #########
        # TO DO : Q2) b Closest phrase to the word clinton :
        #########
        print "Q2) b "
        print model_phrase.most_similar(['clinton'])
        # print model.most_similar(['clinton']) # Just to compare

        #########
        # TO DO : Q2) Closest word to the word Difficult :
        #########
        print "Q3) "
        print model.most_similar(positive=['france', 'berlin'], negative=['germany'])

        print "Q4) "

        print "Similarity btw 'germany' and 'france' is {0}".format(model.similarity('germany','france'))
        print "Similarity btw 'paris' and 'france' is {0}".format(model.similarity('paris', 'france'))
        print "Similarity btw 'berlin' and 'paris' is {0}".format(model.similarity('berlin','paris'))
        print "Similarity btw 'berlin' and 'france' is {0}".format(model.similarity('berlin', 'france'))
        print "Similarity btw 'germany' and 'paris' is {0}".format(model.similarity('germany', 'paris'))
        l = list(model.wv.vocab.keys())
        # Cuisine
        print "Pizza for Italian is {0} for French with score {1}".format(
            model.most_similar(positive=['pizza', 'france'], negative=['italy'])[0][0],
            model.most_similar(positive=['france', 'pizza'], negative=['italy'])[0][1])

        print "Ferrari for Italian is {0} for Germans with score {1}".format(
            model.most_similar(positive=['ferrari', 'germany'], negative=['italy'])[0][0],
            model.most_similar(positive=['ferrari', 'germany'], negative=['italy'])[0][1])


        print "Couscous for Moroccan is {0} for French with score {1}".format(
            model.most_similar(positive=['france', 'couscous'], negative=['morocco'])[0][0],
            model.most_similar(positive=['france', 'couscous'], negative=['morocco'])[0][1])

        l_phrase = list(model_phrase.wv.vocab.keys())
        print "The number of words in model is {0} and the number of phrase in model_phrase is {1}".format(len(l),len(l_phrase))

        print l[:10]
        print l_phrase[:10]
        #print "The phrase of id 777 is: {0}".format(l_phrase[777])
    if False: # THIRD PART
        # Loading model trained on words
        model = word2vec.Word2Vec.load('models/text8.model')
        """
        Sentence embeddings with average(word2vec)
        """
        data_path = 'data/snli.test'
        array_sentences, array_embeddings = avg_word2vec(model, dataset=data_path)

        #########
        # TO DO : do the TODOs in cosine_similarity
        #########
        query_idx = 777  # random sentence
        assert query_idx < len(array_sentences) # little check

        # For the next line to work, you need to implement the "cosine_similarity" function.
        # array_sentences[closest_idx] will be the closest sentence to array_sentences[query_idx].
        #closest_idx = most_similar(query_idx, array_embeddings, array_sentences)

        #########
        # TO DO : Implement the most_5_similar function to output the 5 sentences that are closest to the query.
        # TO DO : Report the 5 most similar sentences to query_idx = 777
        #########
        closest_5_idx = most_5_similar(query_idx, array_embeddings, array_sentences)
        print "Query sentence is sentence of id {0}".format(query_idx)
        print "         {0}".format(array_sentences[query_idx])
        print "     "
        print "The 5 closest sentences are :"
        rank = 1
        for idx in closest_5_idx:
            #########
            # TO DO: Print the 5 most similar sentences to query_idx using closest_5_idx, array_sentences, array_embeddings
            #########
            print "Sentence {0} : id {3} -> {1} \n score: {2}".format(rank,array_sentences[idx], cosine_similarity(array_embeddings[query_idx], array_embeddings[idx]) , idx)
            rank = rank + 1


    if False : # FOURTH PART
        #######
        # Weighted average of word vectors with IDF.
        #######
        data_path = 'data/snli.test'
        model = word2vec.Word2Vec.load('models/text8.model')
        word2idf,wordapp,l = IDF(data_path)
        # Q 1
        #print word2idf['a']
        #print word2idf['the']
        #print word2idf['clinton'] => the word 'clinton' does not appear in documents.

        # Q 2

        array_sentences_idf, array_embeddings_idf = avg_word2vec_idf(model, word2idf, dataset=data_path)

        query_idx = 777  # random sentence
        assert query_idx < len(array_sentences_idf)  # little check

        closest_idx_idf = most_similar(query_idx, array_embeddings_idf, array_sentences_idf)


