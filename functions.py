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


#3 list XML's
import glob
import json
def list_xmls(glob_paths,out_file=None):
    files = []
    for path in glob_paths:
        files.extend(glob.glob(path+"*.tei.xml"))
    print("{} files".format(len(files)))
    for i,f in enumerate(files):
        #replace the "\" returned in the glob paths
        files[i] = f.replace('\\','/')
    
    files.reverse()
    caches = set()
    results = []
    #remove duplicates
    for item in files:
        prefix = item.rsplit('v', 2)[0]
        # check whether prefix already exists
        if prefix not in caches:
            results.append(item)
            caches.add(prefix)
    results.reverse()

    return(results)

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
        print("{} files without duplicates".format(len(texts)),end='\r')
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
def search_texts(texts,search_set,fields,db_name):
    db = client[db_name]
    phrases = db.phrases
    phrases_on_db = phrases.count_documents({})
    if (phrases_on_db == 0):
        print("The database is empty. running your query...")
        matches = run_query(texts,search_set,fields,db_name);
    else:
        valid_ans = bool(False)
        while (not valid_ans):
            redo = str(input("Database already has {} phrases, do you want to retry query? Y-n".format(phrases_on_db))) 
            if (redo in {"Y","y"}):
                matches =run_query(texts,search_set,fields,db_name);
                valid_ans = True
            elif(redo in {"N","n"}):
                cursor = phrases.find({})
                matches = [phrase['phrase'] for phrase in cursor]
                valid_ans = True
            else:   
                print("Please answer with Y,y(Yes) or N,n(No)...")
    print('')
    return(matches)

def run_query(texts,search_set,sections,db_name):
    db = client[db_name]
    phrases = db.phrases
    phrases.drop()
    matches = []
    for text in texts:
        for section in sections:
            if text[section] != None:
                for phrase in sent_tokenize(text[section]):
                    for word in search_set:
                        if (re.search(r"\b" + re.escape(word) + r"\b", phrase.lower())):
                                remove_non_alpha = re.compile('[^a-zA-Z \-]')
                                ratio =  len(remove_non_alpha.sub('',phrase.lower()))/len(phrase.lower())
                                if (ratio>0.8):
                                    try:
                                        lang = detect(phrase)
                                        if (lang=='en'):
                                            match = {"_id":len(matches),"path":text["path"],"phrase":phrase.lower(),"lenght":len(phrase),"match_word":word,"section":section}
                                            phrase_id = phrases.insert_one(match)
                                            #matches.append(Phrase(path=text["path"],phrase=phrase.lower(),lenght=len(phrase)))
                                            matches.append(phrase_id)
                                            print("{} phrase matches".format(len(matches)),end='\r')
                                    except ldError.LangDetectException:
                                        pass  
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

def clean_text(db_name,remove_w=[],stem=False):
    db = client[db_name]
    phrases = db.phrases
    clean_phrases = []
    cursor = phrases.find({})
    rm_w_count =0
    for n,phrase in enumerate(cursor):
        #get phrase text (sentence)
        sentence = phrase["phrase"]

        #remove words from a list.
        for rm_word in remove_w:
            while (re.search(r"\b" + re.escape(rm_word) + r"\b", sentence)):
                rm_w_count = rm_w_count+1
                sentence = re.sub(r"\b" + re.escape(rm_word) + r"\b","", sentence)
            

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
    print("{} palavras removidas".format(rm_w_count))
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

def run_lda(vect,vects,topic_number,db_name):
    db = client[db_name]
    phrases = db.phrases
    topics = db.topics
    X = vects.toarray()
    model = lda.LDA(n_topics=topic_number, n_iter=2500, random_state=1,refresh=10)
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

#populate_DB
import datetime
import ast
import uuid
import timeit

def run_analysis():
    #setup mongo client
    client = MongoClient('localhost', 27017)
    
    #create/access history of 
    hist_db = client.lda_arxiv_log
    history = hist_db.history
    entries = history.count_documents({}) 
    latest = history.find_one({"$query": {}, "$orderby": {"$natural" : -1}})
    if(latest == None):
        print("no Analysis was found in the database... \n we will have to create yours from scratch!")
        this_analysis = input_analysis()
    else:
        valid_ans = bool(False)
        while (not valid_ans):
            redo = input("Do you want to redo an analysis? Y-n")
            if(redo in {"Y","y"}):
                analysis_hist = [hi for hi in history.find({"$query": {}, "$orderby": {"$natural" : -1}})]
                nicknames_list = [hi["nick_name"] for hi in analysis_hist]
                print("these are all the analysis you made:")
                for hi in analysis_hist:
                    print(hi)
                valid_ans = bool(False)
                while (not valid_ans):
                    analysis_old_name = input("please give us the nickname of the analysis you want to redo:")
                    this_analysis = next((hi for hi in analysis_hist if hi["nick_name"] == analysis_old_name), None)
                    if(this_analysis != None):
                        valid_ans = True
                    else:
                        print("please write a valid nickname")
                valid_ans =True
            elif(redo in {"N","n",}):
                this_analysis = input_analysis()
                valid_ans =True
            else:
                print("Please answer with Y,y(Yes) or N,n(No)...")

    glob_paths_ex = ["./pdf/*/"]
    query_words_ex = {"artificial intelligence", "machine learning", "m.l.","a.i."}
    query_sections_ex = {"abstract","text"}
    files = list_xmls(this_analysis["docs_path_list"])
    test_text = create_texts(files)
    test_phrases = search_texts(test_text,this_analysis["query"]["words"],this_analysis["query"]["sections"],this_analysis["db_name"])
    test_clean = clean_text(this_analysis["db_name"],this_analysis["query"]["words"])
    (vect,vects) = create_n_gram(test_clean)
    starttime = timeit.default_timer()
    print(">>> start time is :",starttime)
    model = run_lda(vect,vects,this_analysis["n_topics"],this_analysis["db_name"])
    print("<<< time difference is :", timeit.default_timer() - starttime)
    this_analysis["date"]= datetime.datetime.utcnow()
    this_analysis["_id"]= str(uuid.uuid4())
    history.insert_one(this_analysis)
    print("thanks, your analysis parameters are stored with the nickname: {}.".format(this_analysis["nick_name"]))

def input_analysis():
        hist_db = client.lda_arxiv_log
        history = hist_db.history
        this_analysis ={}
        print("The databadeses in your machine are:")
        dbs_in_machine = []
        for db in client.list_databases():
            print(db)
            dbs_in_machine.append(db["name"])
        database_name = "arxiv_LDA_{}".format(input("Write the name of the new (or old) database: arxiv_LDA_..."))
        this_analysis["db_name"] = database_name
        #get docs_path_list
        valid_ans = bool(False)
        while (not valid_ans):
            try:
                this_analysis["docs_path_list"] = ast.literal_eval(input('Write a list of paths where the documents are found: ["./in", "./this/way", "./pdf/*/"]') or '["./pdf/*/"]' )
                valid_ans = (type(this_analysis["docs_path_list"])==list)
                #check if paths exist...
            except:
                valid_ans = False
            if (valid_ans == False):print("please write your LIST of paths in pythonic syntax")  

        this_analysis["query"] = {}
        valid_ans = bool(False)
        while (not valid_ans):
            try:
                this_analysis["query"]["words"] = ast.literal_eval(input('Write your set of query words: [ "in", "this way"]') or '["artificial intelligence", "machine learning", "m.l.","a.i."]')
                valid_ans = (type(this_analysis["query"]["words"])==list)
            except:
                valid_ans = False
            if (valid_ans == False):print("please write your LIST of words in pythonic syntax") 
        
        valid_ans = bool(False)
        while (not valid_ans):
            try:
                this_analysis["query"]["sections"] = ast.literal_eval(input('Write your set of sections that you want to query: [ "in", "this way"]') or '["abstract","text"]')
                valid_ans = (type(this_analysis["query"]["sections"])==list)
            except:
                valid_ans = False
            if (valid_ans == False):print("please write your LIST of sections in pythonic syntax") 
        
        valid_ans = bool(False)
        while (not valid_ans):
            try:
                this_analysis["n_topics"] = int(input("How many topics do you wish?") or 20)
                valid_ans = True
            except:
                valid_ans = False
            if (valid_ans == False):print("Number of topics must be an Integer")
        
        valid_ans = bool(False)
        nicknames_list = [hi["nick_name"] for hi in history.find({"$query": {}, "$orderby": {"$natural" : -1}})]
        while (not valid_ans):
            this_analysis["nick_name"] = input("wue are almost there... Just give your analysis an nickname:")
            if (this_analysis["nick_name"] not in nicknames_list):
                valid_ans = True
            else:
                print("this nickname already exists please give it an Unique one!")
        return(this_analysis)
