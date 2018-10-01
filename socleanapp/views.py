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
    return render_template("input2.html")


@app.route('/input')
def soclean_input():
    return render_template("input2.html")

@app.route('/about')
def soclean_about():
    return render_template("about.html")


@app.route('/output')
def soclean_output():
  question_title = request.args.get('question_title')
  question_body = request.args.get('question_body')
  question_tags = request.args.get('question_tags').split(",")
  question_tags_unsplit = request.args.get('question_tags')
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
  pans = int(pans*100)
  goodplot = makeplot(goods, plottype = 'good')
  #print("Generated this: ".format(outfile))
  return render_template("output2.html",
                         question_title = question_title,
                         question_body = question_body,
                         question_tags = question_tags,
                         question_tags_unsplit = question_tags_unsplit,
                         pans = pans,
                         badplot = badplot,
                         goodplot = goodplot)


@app.route('/example1')
def example1():
    question_title = "Is there any open-source web service package to view images on remote server like viewing them on local disk?"
    question_body = "I have a remote server, which I cannot SSH directly but I can setup any web service on the server. For my work, I need to frequently view images on the remote server. I expect a web page, where I choose a directory on the server, it automatically show the images files under this directory (not just list the file names). If there are a lot of images, it provides function such as \"previous n images, next n images\" which the value of N can be set. I know it is not very complicated to implement it in php or something like that, but I am not familiar with web service programming. So I am wondering if there is any open source package like this???"
    question_tags = "web, web-services".split(",")
    question_tags_unsplit = "web, web-services"
    myfeatures = extractfeaturevalueslime(question_title, question_body, question_tags)
    encoder = joblib.load('limeencoder.joblib')
    with open('explainer', 'rb') as f:
        explainer = dill.load(f)
    pans, bads, goods = getlime(values = myfeatures, num_features = 11, encoder = encoder, explainer = explainer )
    print("pans = ", pans)
    print("bads = ", bads)
    print("goods = ", goods)
    badplot = makeplot(bads, plottype = 'bad')
    pans = int(pans*100)
    goodplot = makeplot(goods, plottype = 'good')
    return render_template("output2.html",
        question_title = question_title,
        question_body = question_body,
        question_tags = question_tags,
        question_tags_unsplit = question_tags_unsplit,
        pans = pans,
        badplot = badplot,
        goodplot = goodplot)

@app.route('/example2')
def example2():
    question_title = "What are metaclasses in Python?"
    question_body = "What are metaclasses and what do we use them for?"
    question_tags = "python, oop, metaclass, python-datamodel".split(",")
    question_tags_unsplit = "python, oop, metaclass, python-datamodel"
    myfeatures = extractfeaturevalueslime(question_title, question_body, question_tags)
    encoder = joblib.load('limeencoder.joblib')
    with open('explainer', 'rb') as f:
        explainer = dill.load(f)
    pans, bads, goods = getlime(values = myfeatures, num_features = 11, encoder = encoder, explainer = explainer )
    print("pans = ", pans)
    print("bads = ", bads)
    print("goods = ", goods)
    badplot = makeplot(bads, plottype = 'bad')
    pans = int(pans*100)
    goodplot = makeplot(goods, plottype = 'good')
    return render_template("output2.html",
        question_title = question_title,
        question_body = question_body,
        question_tags = question_tags,
        question_tags_unsplit = question_tags_unsplit,
        pans = pans,
        badplot = badplot,
        goodplot = goodplot)
