import numpy as np
from summarizer import SingleModel
import bs4 as bs  
import urllib.request  
import re
import argparse
import nltk
import logging
import heapq


print('Welcome to the Lstm Summarizer!\n')

# choose URL, text file, or copy/paste input
choose_summarizer = input('Enter A for URL input:\nEnter B for text file input:\nEnter C for copy/paste input:\n(case-sensitive)\n')

if choose_summarizer == 'A':
    
    # taking input from website article
    url = input('URL to summarize:\n')
    print('Summarizing...')


    # fetching and reading in data from URL
    scraped_data = urllib.request.urlopen(url)  
    article = scraped_data.read()

    # using beautifulsoup to parse article
    parsed_article = bs.BeautifulSoup(article,'lxml')
    paragraphs = parsed_article.find_all('p')

    # iterating and appending text to string
    article_text = ""

    for p in paragraphs:  
        article_text += p.text

    start = time.time()
    # model default params
    # use bert-base-uncased for smaller, less resource-intensive model
    model = SingleModel(
        model='bert-large-uncased',
        vector_size=None,
        hidden=-2,
        reduce_option='mean'
    )

    end = time.time()
    f_time = end-start
    print(f'Response Time: {f_time}')


    # passing in full text to model
    m = model(article_text)

    # creating final summary with a ratio of 0.13
    summary_file = '\n\nSUMMARY:\n' + m

    # printing summary and full text for comparison
    print(f'\nSUMMARY:\n{model(article_text)}\n')
    print(f'FULL TEXT:\n', article_text)
    
    # appending summary output to text filefilename = 'summary.txt'
    filename = 'summary.txt'

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(filename, append_write) as summary_output:
        for line in summary_file:
            summary_output.write(line)


# text file summary option
elif choose_summarizer == 'B':
    document = input('Enter your <path/to/file.txt> here:\n')

    # reading in text file
    with open(document, 'r') as d:
        text_data = d.read()

    start = time.time()

    # importing model and passing in full text
    model = SingleModel()
    m = model(text_data)

    end = time.time()
    f_time = end-start
    print(f'Response Time: {f_time}')

    # creating final summary with a ratio of 0.13
    summary_file = '\n\nSUMMARY:\n' + m

    # printing summary and full text output for comparison
    print(f'\nSUMMARY:\n{model(text_data)}\n')
    print(f'FULL TEXT:\n', text_data)

    # appending summary output to text file
    filename = 'summary.txt'

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(filename, append_write) as summary_output:
        for line in summary_file:
            summary_output.write(line)

# copy/paste string input option
elif choose_summarizer == 'C':
    text_copy_paste = input('INPUT:\n')
    text_copy_paste = 'Please wait while your summary is processing...'

    start = time.time()

    # importing model and passing in full-text string
    model = SingleModel()
    m = model(text_copy_paste)

    end = time.time()
    f_time = end-start
    print(f'Response Time: {f_time}')

    # creating final summary with a ratio of 0.13
    summary_file = '\n\nSUMMARY:\n' + m

    # printing summary and full text output for comparison
    print(f'\n\nSUMMARY:\n{model(text_copy_paste)}\n')
    print(f'FULL TEXT:\n', text_copy_paste)

    # appending summary output to text file
    filename = 'summary.txt'

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not

    with open(filename, append_write) as summary_output:
        for line in summary_file:
            summary_output.write(line)

else:
    print('\nMust choose from A, B, or C')


