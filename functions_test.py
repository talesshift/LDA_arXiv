#usage of the tests:
# ```
#   from functions import * 
#   test_final('your__testfile_name')
# ```

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
def search_texts(texts,search_set,fields,out_file=None):
    matches = []
    for text in texts:
        for field in fields:
            if text[field] != None:
                if any(word in text[field].lower() for word in search_set):
                    for phrase in text[field].lower().split("."):
                        if any(word in phrase for word in search_set):
                            matches.append({"ID":text["ID"],"phrase":phrase})
                            print(len(matches),end='\r')
    print('')
    if (out_file != None):
        with open('./'+out_file+'.json', 'w') as fout:
            json.dump(matches, fout)
    return(matches)

#5 remove stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd 
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import re

def clean_text(phrases):
    clean_phrases =[]
    for n,phrase in enumerate(phrases):

        regex = re.compile('[^a-zA-Z \-]')
        string_without_punctuation = regex.sub('',phrase['phrase'])
        string_without_punctuation = string_without_punctuation.replace('\r', '')
        string_without_punctuation = string_without_punctuation.replace('\n', '')
        text_tokens = word_tokenize(string_without_punctuation)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        #print(type(tokens_without_sw))
        cleant = " ".join(list(tokens_without_sw))
        clean_phrases.append({"ID":phrase["ID"],"phrase":(cleant)})
        print(str(n)+"-"+str(len(tokens_without_sw)),end='\r')

    return(clean_phrases)
    doc = pd.DataFrame(array)

#5 create n_gram and vocab
from sklearn.feature_extraction.text import CountVectorizer
def create_n_gram(phrases):
    phrase_list = [phrase["phrase"] for phrase in phrases]
    doc = pd.DataFrame(phrase_list)
    doc.columns = ['text']
    vect = CountVectorizer()  
    vects = vect.fit_transform(doc.text)
    return(vect,vects)
    
#clean n_gram (optional)
def clean_n_gram():
    with open('./american-english-huge','r') as f:
            eng_words_huge=set()
            for line in f:
                    strip_lines=line.strip()
                    listli=strip_lines.split()
                    eng_words_huge.add(listli[0])
    
#6 run LDA
import lda 
def run_lda(vects):
    X = vects.toarray()
    model = lda.LDA(n_topics=40, n_iter=2500, random_state=1,refresh=10)
    model.fit(X)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    return(model)

######## TESTS ##########

# variables to EX's
glob_paths_ex = ["./pdf/*/"]

def run_test():
    glob_paths_ex = ["./pdf/*/"]
    test_text = create_texts(list_xmls(glob_paths_ex))
    test_phrases = search_texts(test_text,{"artificial intelligence", "machine learning", " ml "," ai "},{"abstract","text"})
    test_clean = clean_text(test_phrases)
    (vect,vects) = create_n_gram(test_clean)
    model = run_lda(vects)
    return(vect,vects,model)

def print_model(model,vect):
    vocab = vect.get_feature_names()
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 15
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

def get_topic_words(model,vect):
    vocab = vect.get_feature_names()
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 15
    top_topic_words = []
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
        top_topic_words.append(topic_words)
    return(top_topic_words)
        

def phrases_json(model,vect): 
    argamassa = [np.argmax(doc, axis=0) for doc in model.doc_topic_]
    test_phrases = search_texts(create_texts(list_xmls(["./pdf/*/"])),{"artificial intelligence", "machine learning", " ml "," ai "},{"abstract","text"})
    top_words = get_topic_words(model,vect)
    def addict(i,t):
        dicto = test_phrases[i]
        dicto["prob"] = model.doc_topic_[i,t] 
        return(dicto)
    frases =[]
    for topic in range(model.doc_topic_[0].shape[0]):
        frases.append({"topic_id":topic,"top_words":list(top_words[topic]),"phrases":[addict(i,t) for i,t in enumerate(argamassa) if t == topic]})
    return(frases)


def test_final(out_file):
    (vect,vects,model) = run_test()
    topics_json = phrases_json(model,vect)
    if (out_file != None):
        with open('./'+out_file+'.json', 'w') as fout:
            json.dump(topics_json, fout)
    return(vect,vects,model)


#print (type(vects))
#print (len(vect.get_feature_names()))
#print(test_clean[0])
#print(vect.vocabulary_)
#print("")
#(vect,vects,model)=run_test()