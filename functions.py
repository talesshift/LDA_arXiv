#usage of the tests:
# ```
#   from functions import * 
#   test_final('your__testfile_name')
# ```

#mongoDB setup
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
client = MongoClient()
client = MongoClient('localhost', 27017)
db = client.arxiv_LDA_test2_db

#3 list XML's
import glob
import json
def list_xmls(glob_paths,out_file=None):
    files = []
    for path in glob_paths:
        files.extend(glob.glob(path+"*.tei.xml"))
        print(len(files))
    print('')
    for i,f in enumerate(files):
        #replace the "\" returned in the glob paths
        files[i] = f.replace('\\','/')
    if (out_file != None):
        with open('./'+out_file+'.json', 'w') as fout:
            json.dump(files, fout)
    return(files)

#4 create fulltexts array
from contextlib import suppress
import grobid_tei_xml
from os import path

def create_texts(files, out_file=None):
    texts = []
    for file_id,file_path in enumerate(files):

        title = None
        authors = None
        text = None
        doi = None
        citation_count = None
        abstract = None
        
        with suppress(Exception):
            with open(file_path, 'r', encoding="utf8") as d_file:
                doc = grobid_tei_xml.parse_document_xml(d_file.read())
        with suppress(Exception):
            text = doc.body
        with suppress(Exception):
            title= doc.header.title
        with suppress(Exception):
            authors= ", ".join([a.full_name for a in doc.header.authors])
        with suppress(Exception):
            doi= str(doc.header.doi)
        with suppress(Exception):
            citation_count= str(len(doc.citations))
        with suppress(Exception):
            abstract= doc.abstract
        
        texts.append({
            "ID":file_id,
            "path":file_path,
            "text":text,
            "title":title,
            "authors":authors,
            "doi":doi,
            "citation_count":citation_count,
            "abstract":abstract
        })
        print(len(texts),end='\r')
    print('')
    if (out_file != None):
        with open('./'+out_file+'.json', 'w') as fout:
            json.dump(texts, fout)
    return(texts)

#search for texts containing any strings in a set
from nltk.tokenize import sent_tokenize
from collections import namedtuple
from langdetect import detect
import langdetect.lang_detect_exception as ldError
phrases = db.phrases
def search_texts(texts,search_set,fields):
    phrases_on_db = phrases.count_documents({})
    matches = []
    if (phrases_on_db == 0):
        print("The database is empty. running your query...")
        matches = run_query(texts,search_set,fields);
    else:
        valid_ans = bool(False)
        while (not valid_ans):
            redo = str(input("Database already has {} phrases, do you want to retry query? Y-n".format(phrases_on_db))) 
            if (redo in {"Y","y"}):
                matches =run_query(texts,search_set,fields);
                valid_ans = True
            elif(redo in {"N","n"}):
                cursor = phrases.find({})
                matches = [phrase['phrase'] for phrase in cursor]
                valid_ans = True
            else:   
                print("Please answer with Y,y(Yes) or N,n(No)...")
            
    def run_query(texts,search_set,fields):
        phrases.drop()
        for text in texts:
            for field in fields:
                if text[field] != None:
                    for phrase in sent_tokenize(text[field]):
                        for word in search_set:
                            if (re.search(r"\b" + re.escape(word) + r"\b", phrase.lower())):
                                    remove_non_alpha = re.compile('[^a-zA-Z \-]')
                                    ratio =  len(remove_non_alpha.sub('',phrase.lower()))/len(phrase.lower())
                                    if (ratio>0.8):
                                        try:
                                            lang = detect(phrase)
                                            if (lang=='en'):
                                                match = {"_id":len(matches),"path":text["path"],"phrase":phrase.lower(),"lenght":len(phrase),"match_word":word}
                                                phrase_id = phrases.insert_one(match)
                                                #matches.append(Phrase(path=text["path"],phrase=phrase.lower(),lenght=len(phrase)))
                                                matches.append(phrase_id)
                                                print(len(matches),end='\r')
                                        except ldError.LangDetectException:
                                            pass  
        return(matches)
    print('')
    return(matches)

#5 remove stopwords & clean text
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re
remove_non_alpha = re.compile('[^a-zA-Z \-]')

def clean_text(stem=False):
    clean_phrases = []
    cursor = phrases.find({})
    for n,phrase in enumerate(cursor):

        #get phrase text (sentence)
        sentence = phrase["phrase"]

        #remove non_alphanumeric characters
        sentence = remove_non_alpha.sub('',sentence.lower())

        #tokenize sentence
        tokens = word_tokenize(sentence)

        #remove Punctuation (already done)
        #words = [word for word in tokens if word.isalpha()]
        
        #remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [w for w in tokens if not w in stop_words]

        #stem words
        if(stem == True):
            from nltk.stem.porter import PorterStemmer
            porter = PorterStemmer()
            words = [porter.stem(word) for word in tokens]

        #un-tokenize string
        cleant = " ".join(list(words))

        #append clean string to list
        clean_phrases.append(cleant)
        print(str(n)+"-"+str(len(tokens)),end='\r')

    return(clean_phrases)

#5 create n_gram and vocab
from sklearn.feature_extraction.text import CountVectorizer
def create_n_gram(phrases):
    doc = pd.DataFrame(phrases)
    doc.columns = ['text']
    vect = CountVectorizer()  
    vects = vect.fit_transform(doc.text)
    return(vect,vects)
    
#clean n_gram (optional, DOES NOT WORK with porter stemmer)
def clean_n_gram():
    with open('./american-english-huge','r') as f:
            eng_words_huge=set()
            for line in f:
                    strip_lines=line.strip()
                    listli=strip_lines.split()
                    eng_words_huge.add(listli[0])
    
#6 run LDA
import lda 
topics = db.topics
def run_lda(vect,vects):
    X = vects.toarray()
    model = lda.LDA(n_topics=40, n_iter=2500, random_state=1,refresh=10)
    model.fit(X)  # model.fit_transform(X) is also available
    #lista de palavras e probabilidades de cada tópico:
    n_words_on_topic = (model.nzw_ > 0).sum(axis=1)
    top_word_indices_of_topics = [(-model.nzw_[a]).argsort()[:n_words_on_topic[a]] for a in range(len(n_words_on_topic))]
    #save topics to the Db
    topics_on_db = topics.count_documents({})
    for i,top_word_indices in enumerate(top_word_indices_of_topics):
        w_p = [{"word":(vect.get_feature_names())[word],"prob":np.around(model.topic_word_,5)[i][word]} for word in top_word_indices]
        this_topic = {"_id":i,"word_probabilities":w_p}
        if (topics_on_db == 0):
            topics.insert_one(this_topic)
    #add topics to the phrases in the Db
    n_topics_on_document = (model.ndz_ > 0).sum(axis=1)
    top_topic_indices_of_documents = [(-model.ndz_[a]).argsort()[:n_topics_on_document[a]] for a in range(len(n_topics_on_document))]
    for i,top_topic_indices in enumerate(top_topic_indices_of_documents):
        doc_topics = [{"topic": int(topic),"prob":float(np.around(model.doc_topic_,5)[i][topic])} for topic in top_topic_indices]
        phrases.update_one({"_id" :int(i)},{"$set" : {"topics":doc_topics}})
    return(model)
