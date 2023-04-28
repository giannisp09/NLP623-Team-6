import re, string, nltk, itertools

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
import time 
from datetime import datetime


# stop_words = set(stopwords.words("english"))
stop_words = set(stopwords.words('english')) - set(['not'])

#from ekphrasis
plus = set(['th', 'u','nt' ,'uc',' tomorrow', '<url>','<hashtag>', '<repeated>', '<allcaps>','<user>', '<number>', '<money>',  'url','hashtag', 'repeated', 'allcaps','user', 'number', 'money' ])

#time-related
time_related_words = ['today', 'tomorrow','morning', 'night', 'back', 'yesterday', 'year', 'month', 'week', 'day', 'hour', 'minute', 'second', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']



from nltk.stem import PorterStemmer

ps = PorterStemmer()

nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()




def read_data(path_to_csv):
    tweets_df = pd.read_csv(path_to_csv)

    tweets_df['post_date'] = tweets_df['post_date'].map(
       lambda q: unix_date_converted(q,format="%Y-%m-%d", mode="to_date")
    )

    return tweets_df

def df_summary(df):
    print("Data Shape", df.shape)
    print("# of NAs: ", df.isna().sum())


def unix_date_converted(x, format="%Y-%m-%d %H:%M:%S", mode="to_unix"):
    if mode == "to_unix":
      return  time.mktime(datetime.strptime(x, format).timetuple()) 
    else:
      return datetime.fromtimestamp(x).strftime(format)





'''
Ekphrasis Tokenizing

'''

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict



text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

s = 'Profit-Taking Hits Nikkei nt http://t.co/hVWpiDQ1 http://t.co/xJSPwE2z RT @WSJmarkets'
print(" ".join(text_processor.pre_process_doc(s)))


def tokenize(tweet):
  return " ".join(text_processor.pre_process_doc(tweet))

def sentence_tokenize(tweet):
   return " ".join(tokenize(tweet))


import string

print(string.punctuation)

nltk.download("stopwords")
nltk.download('punkt')


   #################################################
  # Definition of the processing methods          #
  #################################################


def uncontract(text): 

    #text = re.sub('user', '', text)
    text = re.sub(r'<url>', '', text)
    text = re.sub(r'<thee>', '', text)
    text = re.sub(r'<hashtag>', '', text)
    text = re.sub(r'<repeated>', '', text)
    text = re.sub(r'<allcaps>', '', text)
    text = re.sub('<user>', '', text)
    text = re.sub('<number>', '', text) 
    #text = re.sub('<money>', '', text) 

    # text = re.sub('user', '', text)
    # text = re.sub('url', '', text)
    # text = re.sub('thee', '', text)
    # text = re.sub('hashtag', '', text)
    # text = re.sub('repeated', '', text)
    # text = re.sub('allcaps', '', text)
    # text = re.sub('user', '', text)
    # text = re.sub('number', '', text) 
    # text = re.sub('money', '', text) 


    text = re.sub(r"(\b)([Aa]re|[Cc]ould|[Dd]id|[Dd]oes|[Dd]o|[Hh]ad|[Hh]as|[Hh]ave|[Ii]s|[Mm]ight|[Mm]ust|[Ss]hould|[Ww]ere|[Ww]ould)n't", r"\1\2 not", text)
    text = re.sub(r"(\b)([Hh]e|[Ii]|[Ss]he|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'ll", r"\1\2 will", text)
    text = re.sub(r"(\b)([Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Yy]ou)'re", r"\1\2 are", text)
    text = re.sub(r"(\b)([Ii]|[Ss]hould|[Tt]hey|[Ww]e|[Ww]hat|[Ww]ho|[Ww]ould|[Yy]ou)'ve", r"\1\2 have", text)
    
    text = re.sub(r"(\b)([Cc]a)n't", r"\1\2n not", text)
    text = re.sub(r"(\b)([Ii])'m", r"\1\2 am", text)
    text = re.sub(r"(\b)([Ll]et)'s", r"\1\2 us", text)
    text = re.sub(r"(\b)([Tt]here)'s", r"\1\2 is", text)
    text = re.sub(r"(\b)([Ww])on't", r"\1\2ill not", text)
    text = re.sub(r"(\b)([Ss])han't", r"\1\2hall not", text)
    text = re.sub(r"(\b)([Yy])(?:'all|a'll)", r"\1\2ou all", text)

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is ", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"it's", " it is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", "  are", text)
    text = re.sub(r"\'d", "  would", text)

    
    #text = re.sub('u', '', text)


    text = re.sub('[0-9]+', '', text)
    #text = re.sub('nt', 'not', text)
    #text = re.sub('th', 'the', text)
    #text = re.sub('rt', '', text)

    

    text = re.sub(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)", "", text)

    






    
    return text

def remove_url_email(tweet):
   #################################################
  # URL regex that starts with `http` or `https`: #
  #################################################

  url_regex_1 = r'^https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$'

  #############################################
  # URL regex that doesn't start with `http`: #
  #############################################

  url_regex_2 = r'^[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)$'

  tweet = re.sub(url_regex_1, 'URL', tweet)
  tweet = re.sub(url_regex_1, 'URL',  tweet)

  email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
 
  tweet = re.sub(email_regex, 'EMAIL', tweet)

  return tweet



def lower(tweet):
  return tweet.lower()

def punctuation(tweet):
  return "".join([char for char in tweet if char not in string.punctuation])

def stemmatize(tweet):
  return " ".join([ps.stem(word) for word in word_tokenize(tweet) ])

def lemmatize(tweet):
  return " ".join([lemmatizer.lemmatize(word) for word in word_tokenize(tweet)])

def stopword_removal(tweet):
  return " ".join([word for word in word_tokenize(tweet) if word not in stop_words])

def ekphrasis_removal(tweet):
  return " ".join([word for word in word_tokenize(tweet) if  word not in plus])

def time_removal(tweet):
  return " ".join([word for word in word_tokenize(tweet) if  word.lower() not in time_related_words])


def preprocess_whole(df):
   df['processed_body'] = df['body'].map(lambda q: tokenize(q))
   df['processed_body'] = df['processed_body'].map(lambda q: uncontract(q))
   df['processed_body'] = df['processed_body'].map(lambda q: remove_url_email(q))
   df['processed_body'] = df['processed_body'].map(lambda q: lower(q))
   df['processed_body'] = df['processed_body'].map(lambda q: punctuation(q))
   df['processed_body'] = df['processed_body'].map(lambda q: stopword_removal(q))
   #df['body'] = df['body'].map(lambda q: stemmatize(q))
   df['processed_body'] = df['processed_body'].map(lambda q: lemmatize(q))
   #df['body'] = df['body'].map(lambda q: stemmatize(q))

   return df

