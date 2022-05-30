######## TESTS ##########
import functions as ft

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

#OLD CLEANING
    #regex = re.compile('[^a-zA-Z \-]')
    #string_without_punctuation = regex.sub('',phrase["phrase"])
    #string_without_punctuation = string_without_punctuation.replace('\r', '')
    #string_without_punctuation = string_without_punctuation.replace('\n', '')
    #text_tokens = word_tokenize(string_without_punctuation)
    #tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    #print(type(tokens_without_sw))