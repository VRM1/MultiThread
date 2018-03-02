'''
This program is for pre-processing amazon revies data
and create a doc2vec using gensim. 
'''

'''
1. open the dataset
2. remove stop words, and apply lematization
Author: Vineeth
'''
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict, Counter
from tqdm import tqdm
from os import listdir
import multiprocessing
import os
import numpy as np
import json
import nltk
from functools import partial
from SplitFiles import split_file
import operator
import gensim
'''
A generic function to process all textual data
'''
loc = 'dataset/'
# a count dictionary to hold the selected top K words
lukup_wrds=Counter()
def GetProcessed(dname):
    # tokenizes the sentence by considering only alpha numeric characters
    tokenizer = RegexpTokenizer(r'\w+')
    with open(loc+dname,'r') as fp:
        container=defaultdict(list)
        wrd_cnt = Counter()
        for line in tqdm(fp):
            # get the textual contents. This depends on the structure of data
            contens = line.split('\t')
            tokens = tokenizer.tokenize(contens[1])
            downcased = [x.lower() for x in tokens]
            for x in downcased:
                wrd_cnt[x] += 1
                container[contens[0]].append(x)
        return (container,wrd_cnt)

# this method removes the words in reviews that donot meet the frequency criteria
def RemWrdInRev(review,lukup):
     
    itm = review[0]
#     print 'key:{},value:{}'.format(itm,wrd)
    new_revs = [wrd for wrd in review[1] if wrd in lukup]
    return (itm,new_revs)

    
def CleanData(rev_name,grph_name,lop):
    
    
    split_file('dataset/'+rev_name+'.txt', lines_per_file=lop)
    mypath = 'dataset/'
    itm_revs=defaultdict(list)
    wrd_cnt = Counter()
    tmp_cnt=[]
    #Context window length
    context_size = 4
    #Seed for the RNG, to make the result reproducible
    seed = 1
    # get the max number of features per document
    num_features = 300
    # minimum number of words to consider
    min_word_count = 15
    graph_data = set()
    with open('dataset/'+grph_name+'.json','r') as fp:
        itm_pairs = json.load(fp)
    for itms in itm_pairs:
        tmp_pairs = [i.strip() for i in itms.strip('(|)').split(',')]
        graph_data.update(tmp_pairs)
    # create threads equal to the number of cpus available
    num_workers = multiprocessing.cpu_count()
    p = multiprocessing.Pool(num_workers)
    filenames = [f for f in listdir(mypath) if f.startswith(rev_name+'_')]
    for result in p.imap(GetProcessed, filenames):
        itms,wrds = result[0],result[1]
        for i in itms:
            # check if the review is a part of the graph data
            if i in graph_data:
                itm_revs[i] += itms[i]
        for w in wrds:
            wrd_cnt[w] += wrds[w]
    # now iterate through the the graph of item pairs and remove those item for which there is no review
    pair_keys = itm_pairs.keys()
    for itms in pair_keys:
        tmp_pairs = [i.strip() for i in itms.strip('(|)').split(',')]
        itm_a,itm_b = tmp_pairs[0],tmp_pairs[1]
        if itm_a not in itm_revs or itm_b not in itm_revs:
            del itm_pairs[itms]
    print 'total item reviews: {}'.format(len(itm_revs))
    print 'total item pair relations: {}'.format(len(itm_pairs))
    wrd_cnt = sorted(wrd_cnt.items(), key=operator.itemgetter(1), reverse=True)
    for w,cnt in wrd_cnt:
            tmp_cnt.append(int(cnt))
    print 'Before Filtration, # unique words: {}, median wrd count: {}, avg wrd count: {}' \
            .format(len(wrd_cnt),np.median(tmp_cnt),np.average(tmp_cnt))
    # create a tagged document that is required by the gensim model
    documents = [gensim.models.doc2vec.TaggedDocument(itm_revs[i],[i]) for i in itm_revs]
    print documents[0]
    # initialize the doctovec model
    doc2vec_model = gensim.models.doc2vec.Doc2Vec( seed=seed, workers=num_workers,  vector_size=num_features, \
                                                   min_count=min_word_count, window=context_size)
   
    doc2vec_model.build_vocab(documents)
    print("The vocabword2vec_model.iterulary is built")
    print("Word2Vec vocabulary length: ", len(doc2vec_model.wv.vocab))
    
    #Start training the model
    doc2vec_model.train(documents=documents,total_examples=doc2vec_model.corpus_count,epochs=doc2vec_model.iter)
    
    #Save the model
    doc2vec_model.save("output/"+rev_name+".d2v")
    print("Model saved")
    # write the new graph
    with open('output/'+grph_name+'_filtered.json','w') as fp:
        json.dump(itm_pairs,fp)
    
    # cleanup the split files
    for f in listdir(mypath):
        if f.startswith(rev_name+'_'):
            os.remove(mypath+f)
    
if __name__ == '__main__':
    
    lines_per_file = 20000

    dbs =[['Clothing_Shoes_and_Jewelry_reviews','Men_Clothing_graph_4class'],\
          ['Clothing_Shoes_and_Jewelry_reviews','Women_Clothing_graph_4class'],\
          ['Movies_and_TV_reviews','Movies_graph_4class'],\
          ['Musical_Instrument_reviews','Musical_Instrument_graph_4class'],\
          ['Electronics_reviews','Electronics_graph_2class'],\
          ['Books_reviews','Books_graph_4class']]
    dbname=dbs[5][0]
    grph_name = dbs[5][1]
    CleanData(dbname,grph_name,lines_per_file)
