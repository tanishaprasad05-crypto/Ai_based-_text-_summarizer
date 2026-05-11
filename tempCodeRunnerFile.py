"""
AI Based Text Summarizer  ·  v2.0
Real-world AI/NLP Portfolio Project
New: Sentence slider · Copy buttons · Keyword extractor · Readability score
     Session history · Download as .txt · Sentiment indicator · Teal/cyan theme
"""

import streamlit as st
import nltk
import time
import io
import re
import math
import collections
import datetime

import PyPDF2
from transformers import pipeline
from sumy.summarizers.lsa import LsaSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)