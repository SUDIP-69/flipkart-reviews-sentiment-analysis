import re
import os
import nltk
import joblib
import requests
import numpy as np
import random
from bs4 import BeautifulSoup
import urllib.request as urllib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
from flask import Flask, render_template, request
import time

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# word_2_int = joblib.load('word2int.sav')
# model = joblib.load('sentiment.sav')
# stop_words = set(open('stopwords.txt'))

# Set headers to avoid bot detection
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

def clean(x):
    x = re.sub(r'[^a-zA-Z ]', ' ', x)  # replace everything that's not an alphabet with a space
    x = re.sub(r'\s+', ' ', x)  # replace multiple spaces with one space
    x = re.sub(r'READ MORE', '', x)  # remove READ MORE
    x = x.lower()
    x = x.split()
    y = []
    for i in x:
        if len(i) >= 3:
            if i == 'osm':
                y.append('awesome')
            elif i == 'nyc':
                y.append('nice')
            elif i == 'thanku':
                y.append('thanks')
            elif i == 'superb':
                y.append('super')
            else:
                y.append(i)
    return ' '.join(y)

def extract_all_reviews(url, clean_reviews, org_reviews, customernames, commentheads, ratings):
    try:
        request = urllib.Request(url, headers=HEADERS)
        with urllib.urlopen(request) as u:
            page = u.read()
            page_html = BeautifulSoup(page, "html.parser")

        # Check for CAPTCHA
        if "captcha" in page_html.get_text().lower():
            print("CAPTCHA detected! Cannot scrape this page.")
            return False

        reviews = page_html.find_all('div', {'class': 't-ZTKy'})
        commentheads_ = page_html.find_all('p', {'class': '_2-N8zT'})
        customernames_ = page_html.find_all('p', {'class': '_2sc7ZR _2V5EHH'})
        ratings_ = page_html.find_all('div', {'class': ['_3LWZlK _1BLPMq', '_3LWZlK _32lA32 _1BLPMq', '_3LWZlK _1rdVr6 _1BLPMq']})

        for review in reviews:
            x = review.get_text()
            org_reviews.append(re.sub(r'READ MORE', '', x))
            clean_reviews.append(clean(x))

        for cn in customernames_:
            customernames.append('~' + cn.get_text())

        for ch in commentheads_:
            commentheads.append(ch.get_text())

        ra = []
        for r in ratings_:
            try:
                if int(r.get_text()) in [1, 2, 3, 4, 5]:
                    ra.append(int(r.get_text()))
                else:
                    ra.append(0)
            except:
                ra.append(r.get_text())
        ratings += ra

        return True
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return False

def tokenizer(s):
    s = s.lower()  # convert the string to lower case
    tokens = nltk.tokenize.word_tokenize(s)  # tokenize the string
    tokens = [t for t in tokens if len(t) > 2]  # remove short tokens
    tokens = [t for t in tokens if t not in stop_words]  # remove stop words
    return tokens

def tokens_2_vectors(token):
    X = np.zeros(len(word_2_int) + 1)
    for t in token:
        if t in word_2_int:
            index = word_2_int[t]
        else:
            index = 0
        X[index] += 1
    X = X / X.sum()
    return X

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results', methods=['GET'])
def result():
    url = request.args.get('url')
    nreviews = int(request.args.get('num'))
    clean_reviews = []
    org_reviews = []
    customernames = []
    commentheads = []
    ratings = []

    # Fetch the product page
    request = urllib.Request(url, headers=HEADERS)
    with urllib.urlopen(request) as u:
        page = u.read()
        page_html = BeautifulSoup(page, "html.parser")

    proname = page_html.find_all('span', {'class': 'B_NuCI'})[0].get_text()
    price = page_html.find_all('div', {'class': '_30jeq3 _16Jk6d'})[0].get_text()

    # Get link to all reviews
    all_reviews_url = page_html.find_all('div', {'class': 'col JOpGWq'})[0]
    all_reviews_url = all_reviews_url.find_all('a')[-1]
    all_reviews_url = 'https://www.flipkart.com' + all_reviews_url.get('href')
    url2 = all_reviews_url + '&page=1'

    # Scrape reviews page by page
    while True:
        x = len(clean_reviews)
        success = extract_all_reviews(url2, clean_reviews, org_reviews, customernames, commentheads, ratings)
        if not success or len(clean_reviews) >= nreviews:
            break

        # Add random delay to avoid getting blocked
        time.sleep(random.uniform(1, 5))

        # Go to the next page
        url2 = url2[:-1] + str(int(url2[-1]) + 1)

    # Limit the number of reviews
    org_reviews = org_reviews[:nreviews]
    clean_reviews = clean_reviews[:nreviews]
    customernames = customernames[:nreviews]
    commentheads = commentheads[:nreviews]
    ratings = ratings[:nreviews]

    # Generate word cloud
    for_wc = ' '.join(clean_reviews)
    wcstops = set(STOPWORDS)
    wc = WordCloud(width=1400, height=800, stopwords=wcstops, background_color='white').generate(for_wc)
    plt.figure(figsize=(20, 10), facecolor='k', edgecolor='k')
    plt.imshow(wc, interpolation='bicubic')
    plt.axis('off')
    plt.tight_layout()
    CleanCache(directory='static/images')
    plt.savefig('static/images/woc.png')
    plt.close()

    # Analyze sentiment based on stars
    d = []
    for i in range(len(org_reviews)):
        x = {
            'review': org_reviews[i],
            'cn': customernames[i],
            'ch': commentheads[i],
            'stars': ratings[i]
        }
        if ratings[i] != 0:
            if ratings[i] in [1, 2]:
                x['sent'] = 'NEGATIVE'
            else:
                x['sent'] = 'POSITIVE'
        d.append(x)

    np, nn = 0, 0
    for i in d:
        if i['sent'] == 'NEGATIVE':
            nn += 1
        else:
            np += 1

    return render_template('result.html', dic=d, n=len(clean_reviews), nn=nn, np=np, proname=proname, price=price)

@app.route('/wc')
def wc():
    return render_template('wc.html')

class CleanCache:
    def __init__(self, directory=None):
        self.clean_path = directory
        if os.listdir(self.clean_path) != list():
            files = os.listdir(self.clean_path)
            for fileName in files:
                print(fileName)
                os.remove(os.path.join(self.clean_path, fileName))
        print("Cleaned!")

if __name__ == '__main__':
    app.run(debug=True)
