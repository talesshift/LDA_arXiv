import pymongo
from pymongo import MongoClient
import numpy as np
import os
client = MongoClient('localhost', 27017)
for database in client.list_databases():
    print(database)
db_names = ['arxiv_LDA_5t_hyph','arxiv_LDA_10t_hyph','arxiv_LDA_20t_hyph','arxiv_LDA_40t_hyph']
dbs = [client[db_name] for db_name in db_names]
counts = [db.topics.count_documents({}) for db in dbs]
phrases = [{phrase['a_id']:phrase['topics'][0]['topic'] for phrase in db.phrases.find()} for db in dbs]
def conectionsAB(a,b):
    matrix = np.zeros((counts[a],counts[b]))
    for phrase, value in phrases[a].items():
        try:
            matrix[value,phrases[b][phrase]] = matrix[value,phrases[b][phrase]]+1
        except:
            #print(value,phrase)
            pass #why some phrases are not paired?
    labelA = ["A{}".format(n) for n in range(counts[a])]
    labelB = ["B{}".format(n) for n in range(counts[b])]
    label = labelA + labelB
    #print(matrix)
    sources = []
    targets = []
    values = []
    cont_src = 0
    for src in matrix:
        cont_trg = counts[a]
        for trg in src:
            sources.append(cont_src)
            targets.append(cont_trg)
            values.append(trg)
            cont_trg = cont_trg+1
        cont_src = cont_src+1
    return(sources,targets,values,label)
sankeys = [conectionsAB(n,n+1) for n in range(len(counts)-1)]
parameters = list(map(list, zip(*sankeys)))
sources_t = parameters[0]
targets_t = parameters[1]
values_t = parameters[2]
label_t = parameters [3]

import plotly.graph_objects as go

for i in range(len(sankeys)):
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = label_t[i],
          color = "blue"
        ),
        link = dict(
          source = sources_t[i], # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = targets_t[i],
          value = values_t[i]
      ))])

    fig.update_layout(title_text="Sankey Diagram: from {} topics to {} topics".format(counts[i],counts[i+1]), font_size=10)
    fig.show()
