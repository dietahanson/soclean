#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import uuid
import pickle
import numpy as np
from textatistic import Textatistic, punct_clean
from sklearn.externals import joblib
import dill
import imgkit
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')
from io import BytesIO


with open('top2017tags.pkl', 'rb') as f:
    top20 = pickle.load(f)

limefilename = 'limerf_model.sav'
loaded_limemodel = pickle.load(open(limefilename, 'rb'))

encoder = joblib.load('limeencoder.joblib')

with open('explainer2', 'rb') as f:
   explainer = dill.load(f)


dummerdict = {'Friday':0,
          'Monday':1,
          'Saturday':2,
          'Sunday':3,
          'Thursday':4,
          'Tuesday':5,
          'Wednesday':6}



def extractfeaturevalueslime(title, body, tags):
    tagnum = len(tags)
    titleuppers = len(re.findall(r'[A-Z]', title))
    titlelength = len(title)
    titleqmarks = len(re.findall(r'\?',title))
    snippetslist = body.split("code>")[1::2]
    cleansnippets = [code.replace("</", "") for code in snippetslist]
    nsnippets = len(cleansnippets)
    bodychunks = body.split("code>")[0::2]
    cleanbodychunks = [re.sub('(<[^>]+>)|(\\n)|(\\r)|(<)', '', chunk) for chunk in bodychunks]
    conbodylength = len(" ".join(cleanbodychunks))
    # get readability score for body
    try:
        clean = punct_clean(",".join(cleanbodychunks)+".")
        read = Textatistic(clean).flesch_score
    except:
        read = -1000
    snippetlength = len("".join(cleansnippets))
    today = time.strftime("%A")
    creationday = dummerdict[today]
    popcount = np.sum([t in top20 for t in tags])
    bodyqmarks = len(re.findall(r'\?',(",".join(cleanbodychunks))))
    values = np.array([tagnum,titleuppers,titlelength,titleqmarks,nsnippets,
    conbodylength,read,snippetlength,creationday,popcount,bodyqmarks]).astype(float)
    return(values)


def getlime(values, num_features, encoder, explainer):
    predict_fn = lambda x: loaded_limemodel.predict_proba(encoder.transform(x.reshape(-1,11))).astype(float)
    exp = explainer.explain_instance(values,
                                 predict_fn, num_features=num_features)
    reasonlist = exp.as_list()
    pans = exp.local_pred[0]
    #explainhtml = exp.save_to_file('limeout.html')
    #random_text = str(uuid.uuid1())[:6]
    #filename = "limepic2_" + random_text + ".jpg"
    #print("FILENAME is " + filename)
    #imgkit.from_file('limeout.html', 'socleanapp/static/{}'.format(filename))
    bads = pd.DataFrame.from_dict(exp.as_list())
    bads.columns = ['feature', 'weight']
    bads = bads[ bads.weight < 0 ].sort_values('weight')
    bads = bads[0:5]
    goods = pd.DataFrame.from_dict(exp.as_list())
    goods.columns = ['feature', 'weight']
    goods = goods[ goods.weight > 0 ].sort_values('weight', ascending = False)
    goods = goods[0:5]
    return pans, bads, goods

def makeplot(df, plottype):
    names = list(df['feature'])
    scores = list(df['weight'])
    y = range(5,5-len(names),-1) # TODO(dieta) think about what to do here.
    fig, ax = plt.subplots(figsize=(2.5,4))
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    colors = plt.cm.RdYlGn((np.sign(scores)+1)/2)
    if plottype == "good":
        ax.yaxis.set_ticks_position('right')
    plt.barh(y,scores,color=colors)
    plt.yticks(y, names, fontsize=16)

    try:
        print("creating figure file")
        figfile = BytesIO()
        plt.savefig(figfile, format='svg', bbox_inches='tight')
        # Add plt.close() to avoid Assertion failed:
        # (NSViewIsCurrentlyBuildingLayerTreeForDisplay() != currentlyBuildingLayerTree)
        # errors, following https://stackoverflow.com/questions/49286741/matplotlib-not-working-with-python-2-7-and-django-on-osx
        plt.close()
        figfile.seek(0)
        figdata_svg = b'<svg' + figfile.getvalue().split(b'<svg')[1]
        return figdata_svg.decode('utf-8')
    except err:
        print("exception", err)
        return None
