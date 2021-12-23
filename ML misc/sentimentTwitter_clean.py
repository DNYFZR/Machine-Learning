#Twitter Data Sentiment Analysis (NLP)
#Link: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk

#Imports & downloads
import nltk, re, string, random

from nltk import FreqDist, classify, NaiveBayesClassifier as nbc
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag #Types link: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#Downloads if required
#1. nltk.download('twitter_samples') 
#2. nltk.download('punkt') 
#3. nltk.download('wordnet')
#4. nltk.download('averaged_perceptron_tagger')
#5. nltk.download('stopwords')

#Functions
#1. Normalise & noise removal - Hyperlinks, Twitter handles & punctuation / special characters (context loss issues)
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

 #2. Generator function
def get_all(clean_list):
    for tokens in clean_list:
        for token in tokens:
            yield token

#3. Build dictionary
def get_tweets(clean_list):
    for tweet_t in clean_list:
        yield dict([token, True] for token in tweet_t)

#NLP Twitter Data Modelling
if __name__ == "__main__":

    # Tokenizing Data
    p_tweets = twitter_samples.strings('positive_tweets.json')
    n_tweets = twitter_samples.strings('negative_tweets.json')
    text = twitter_samples.strings('tweets.20150430-223406.json')
    tt = twitter_samples.tokenized('positive_tweets.json')

    stop_words = stopwords.words('english')

    p_tokens = twitter_samples.tokenized('positive_tweets.json')
    n_tokens = twitter_samples.tokenized('negative_tweets.json')

    p_cleanT = []
    n_cleanT = []

    for tokens in p_tokens:
        p_cleanT.append(remove_noise(tokens,stop_words))
    for tokens in n_tokens:
        n_cleanT.append(remove_noise(tokens, stop_words))

    #Determine Word Density
    all_pwords = get_all(p_cleanT)
    all_nwords = get_all(n_cleanT)

    #Frequency dist analysis
    fd_p = FreqDist(all_pwords)
    fd_n = FreqDist(all_nwords)

    print('Top 5 positive:', fd_p.most_common(5))
    print('Top 5 negative:', fd_n.most_common(5))

    #Data Preparation
    pt_model = get_tweets(p_cleanT)
    nt_model = get_tweets(n_cleanT)

    #Split data for training / testing
    p_dataset = [(tweet_dict, 'Positive') for tweet_dict in pt_model]
    n_dataset = [(tweet_dict, 'Negative') for tweet_dict in nt_model]

    dataset = p_dataset + n_dataset

    #Shuffle to randomise samples
    random.shuffle(dataset)

    train_data = dataset[:5000]
    test_data = dataset[5001:]

    #Build & Test 
    classifier = nbc.train(train_data)
    print('Model Accuracy:', round(100* classify.accuracy(classifier, test_data),1),'%')
    print(classifier.show_most_informative_features(20))

    #Random testing
    custom_1 = 'I just ordered once from TerribleCo, they screwed up, never used the app again'
    custom_2 = 'Congrats #SportStar on your 7th best goal from last season winning goal of the year :) #Baller #Topbin #oneofmanyworldies'
    custom_3 = 'PowerCorp loves wild estimates, so imaginative'

    custom_t1 = remove_noise(word_tokenize(custom_1))
    custom_t2 = remove_noise(word_tokenize(custom_2))
    custom_t3 = remove_noise(word_tokenize(custom_3))

    print(classifier.classify(dict([token, True] for token in custom_t1)), ':', custom_1)
    print(classifier.classify(dict([token, True] for token in custom_t2)), ':', custom_2)
    print(classifier.classify(dict([token, True] for token in custom_t3)), ':', custom_3)
