#Twitter Data Sentiment Analysis (NLP)
#Link: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk

#Imports & downloads
import nltk
#nltk.download('twitter_samples') #Data
from nltk.corpus import twitter_samples

#Tokenizing Data
#nltk.download('punkt') #Tokenizing model

p_tweets = twitter_samples.strings('positive_tweets.json')
n_tweets = twitter_samples.strings('negative_tweets.json')
text = twitter_samples.strings('tweets.20150430-223406.json')
tt = twitter_samples.tokenized('positive_tweets.json')

#Normalise the data
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag #Types link: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
from nltk.stem.wordnet import WordNetLemmatizer

def lem_sent(tokens):
    lem = WordNetLemmatizer()
    lemd_sent = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemd_sent.append(lem.lemmatize(word, pos))
    return lemd_sent

#Noise removal
    #Hyperlinks, Twitter handles & punctuation / special characters (context loss issues)
import re, string
#nltk.download('stopwords')
from nltk.corpus import stopwords

def remove_noise(tt, stop_words = ()):
    clean_t = []

    for token, tag in pos_tag(tt):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lem = WordNetLemmatizer()
        token = lem.lemmatize(token, pos)

        if len(token) >0 and token not in string.punctuation and token.lower() not in stop_words:
            clean_t.append(token.lower())
    return clean_t

stop_words = stopwords.words('english')

#print(remove_noise(tt[0], stop_words))

p_tokens = twitter_samples.tokenized('positive_tweets.json')
n_tokens = twitter_samples.tokenized('negative_tweets.json')

p_cleanT = []
n_cleanT = []

for tokens in p_tokens:
    p_cleanT.append(remove_noise(tokens,stop_words))
for tokens in n_tokens:
    n_cleanT.append(remove_noise(tokens, stop_words))

#print(p_tokens[500], p_cleanT[500])

#Determine Word Density
#Generator function
def get_all(clean_list):
    for tokens in clean_list:
        for token in tokens:
            yield token

all_pwords = get_all(p_cleanT)
all_nwords = get_all(n_cleanT)

#Frequency dist analysis
from nltk import FreqDist

fd_p = FreqDist(all_pwords)
fd_n = FreqDist(all_nwords)

#print(fd_p.most_common(5), fd_n.most_common(5))

#Data Preparation

#Build dictionary
def get_tweets(clean_list):
    for tweet_t in clean_list:
        yield dict([token, True] for token in tweet_t)

pt_model = get_tweets(p_cleanT)
nt_model = get_tweets(n_cleanT)

#Split data for training / testing
import random
p_dataset = [(tweet_dict, 'Positive') for tweet_dict in pt_model]
n_dataset = [(tweet_dict, 'Negative') for tweet_dict in nt_model]

dataset = p_dataset + n_dataset

#Shuffle to randomise samples
random.shuffle(dataset)

train_data = dataset[:7000]
test_data = dataset[7000:]

#Build & Test 
from nltk import classify
from nltk import NaiveBayesClassifier as nbc
classifier = nbc.train(train_data)

print('Classifier accuracy:', classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(5))

#Random testing
from nltk.tokenize import word_tokenize
custom = 'I just ordered once from TerribleCo, they screwed up, never used the app again'
custom_2 = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'
custom_3 = 'Thank you for sending my baggage to CityX and flying me to CityY at the same time. Brilliant service. #thanksGenericAirline'

custom_t = remove_noise(word_tokenize(custom))
custom_t2 = remove_noise(word_tokenize(custom_2))
custom_t3 = remove_noise(word_tokenize(custom_3))

print(classifier.classify(dict([token, True] for token in custom_t)), ':', custom)
print(classifier.classify(dict([token, True] for token in custom_t2)), ':', custom_2)
print(classifier.classify(dict([token, True] for token in custom_t3)), ':', custom_3)