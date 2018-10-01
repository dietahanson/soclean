#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import render_template
from socleanapp import app
import pandas as pd
from flask import request
from myModel import extractfeaturevalueslime, getlime, makeplot
import numpy as np
import re
from pandas import DataFrame
from textatistic import Textatistic
import time
import pickle
from sklearn.externals import joblib
import dill
#from PIL import Image
import imgkit
from flask import url_for




@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Margot' },
       )


@app.route('/input')
def soclean_input():
    return render_template("input.html")

@app.route('/output')
def soclean_output():
  question_title = request.args.get('question_title')
  question_body = request.args.get('question_body')
  question_tags = request.args.get('question_tags').split(",")
  myfeatures = extractfeaturevalueslime(question_title,
                                    question_body, question_tags)
  encoder = joblib.load('limeencoder.joblib')
  with open('explainer', 'rb') as f:
     explainer = dill.load(f)
  pans, bads, goods = getlime(values = myfeatures, num_features = 11, encoder = encoder, explainer = explainer )
  print("pans = ", pans)
  print("bads = ", bads)
  print("goods = ", goods)
  badplot = makeplot(bads, plottype = 'bad')
  #goodplot = makeplot(goods, plottype = 'good')
  #print("Generated this: ".format(outfile))
  return render_template("output.html",
                         question_title = question_title,
                         question_body = question_body,
                         question_tags = question_tags,
                         pans = pans,
                         badplot = badplot)
                         #goodplot = "static/{}".format(goodplot))
