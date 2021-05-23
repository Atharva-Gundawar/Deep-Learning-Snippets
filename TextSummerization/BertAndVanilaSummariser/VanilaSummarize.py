import numpy as np
from summarizer import SingleModel
import bs4 as bs  
import urllib.request  
import re
import argparse
import nltk
import logging
import heapq
import time

print('Welcome to the Baseline Model Summarizer!\n')

# choose URL, text file, or string input
choose_input = input('Enter A for URL input:\nEnter B for text file input:\nEnter C for copy/paste input:\n(case-sensitive)\n')

if choose_input == 'A':
    # enter URL
    url = input('URL to summarize: \n')
    print('Summarizing...')
    start = time.time()

    # fetching and reading in data from URL
    scraped_data = urllib.request.urlopen(url)  
    article = scraped_data.read()

    # using beautifulsoup to parse article
    parsed_article = bs.BeautifulSoup(article,'lxml')
    paragraphs = parsed_article.find_all('p')

    # iterating and appending to full-text string
    article_text = ""

    for p in paragraphs:  
        article_text += p.text

    # text clean up
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)  
    article_text = re.sub(r'\s+', ' ', article_text)  

    processed_article = re.sub('[^a-zA-Z]', ' ', article_text )  
    processed_article = re.sub(r'\s+', ' ', processed_article)

    # sentence-level tokenization of full text
    sentence_list = nltk.sent_tokenize(article_text)  

    # NLTK stopwords
    stopwords = nltk.corpus.stopwords.words('english')

    # creating term frequency dict
    word_frequencies = {}  
    for word in nltk.word_tokenize(processed_article):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    # adding term frequency ratio as dict values
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    # ranking sentences for summary inclusion
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # creating final summary with default 4 highest-scoring sentences
    summary_sentences = heapq.nlargest(4, sentence_scores, key=sentence_scores.get)
    summary_sentences = ''.join(summary_sentences)
    summary_file = '\n\nSUMMARY:\n' + summary_sentences

    end = time.time()
    f_time = end-start
    print(f'Response Time: {f_time}')


    # printing summary and full text for comparison
    print(f'\nSUMMARY:\n{summary_sentences}\n\n')
    print(f'FULL TEXT:')
    print(article_text)

    # appending summary output to text file
    filename = 'summary.txt'

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(filename, append_write) as summary_output:
        for line in summary_file:
            summary_output.write(line)

# text file input option
elif choose_input == 'B':

    document = input('Please enter your <path/to/file.txt> here:\n')

    # reading in text file
    with open(document, 'r') as d:
        text_data = d.read()
    
    start = time.time()

    # text clean up
    text_data = re.sub(r'\[[0-9]*\]', ' ', text_data)  
    text_data = re.sub(r'\s+', ' ', text_data)  

    processed_article = re.sub('[^a-zA-Z]', ' ', text_data )  
    processed_article = re.sub(r'\s+', ' ', processed_article)

    # sentence-level tokenization of full text
    sentence_list = nltk.sent_tokenize(text_data)  

    # NLTK stopword list
    stopwords = nltk.corpus.stopwords.words('english')

    # creating term frequency dict
    word_frequencies = {}  
    for word in nltk.word_tokenize(processed_article):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    # adding term frequency ratios as dict values
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    # ranking sentences for summary inclusion
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    # creating final summary with default 4 highest-scoring sentences
    summary_sentences = heapq.nlargest(4, sentence_scores, key=sentence_scores.get)
    summary_sentences = ''.join(summary_sentences)
    summary_file = '\n\nSUMMARY:\n' + summary_sentences

    end = time.time()
    f_time = end-start
    print(f'Response Time: {f_time}')

    # printing summary and full-text output for comparison
    print(f'\nSUMMARY:\n{summary_sentences}\n')
    print(f'FULL TEXT:')
    print(text_data)

    # appending summary to text file
    filename = 'summary.txt'

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(filename, append_write) as summary_output:
        for line in summary_file:
            summary_output.write(line)


# copy/paste string input option
elif choose_input == 'C':
    print("\nFor Option C, simply copy/paste any text you'd like to summarize below.\n\n")
    print("If you'd like a copy of your summary, you can find it in the `your_summaries` directory.\n")

    # reading in text as string
    text_copy_paste = input('INPUT:\n')
    text_copy_paste = str(text_copy_paste)

    start = time.time()

    # text processing and clean up
    text_copy_paste = re.sub(r'\[[0-9]*\]', ' ', text_copy_paste)  
    text_copy_paste = re.sub(r'\s+', ' ', text_copy_paste)  

    processed_article = re.sub('[^a-zA-Z]', ' ', text_copy_paste )  
    processed_article = re.sub(r'\s+', ' ', processed_article)

    # sentence-level tokenization of full text
    sentence_list = nltk.sent_tokenize(text_copy_paste)  

    # NLTK stopword list; optionally can use sklearn stopwords
    stopwords = nltk.corpus.stopwords.words('english')

    # creating term frequency dict
    word_frequencies = {}  
    for word in nltk.word_tokenize(processed_article):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())
    
    # adding term frequency ratios as dict values
    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

    # final ranking for summary sentence inclusion
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    end = time.time()
    f_time = end-start
    print(f'Response Time: {f_time}')

    # creating final summary with default 4 highest-scoring sentences
    summary_sentences = heapq.nlargest(4, sentence_scores, key=sentence_scores.get)
    summary_sentences = ''.join(summary_sentences)
    summary_file = '\n\nSUMMARY:\n' + summary_sentences

    # printing summary and full-text output for comparison
    print(f'\nSUMMARY:\n{summary_sentences}\n')
    print(f'FULL TEXT:')
    print(text_copy_paste)

    # appending summary to text file
    filename = 'summary.txt'

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(filename, append_write) as summary_output:
        for line in summary_file:
            summary_output.write(line)
