#IMPORTS
import string
import nltk
nltk.download('vader_lexicon')
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pdf
import spacy
import pandas as pd

def model():
    emails = pd.read_csv('resources/zPDFS/emails.csv')
    emails = emails.drop_duplicates()
    emails['length'] = emails['text'].apply(len)
    emails['punctuation'] = emails['text'].apply(lambda x: sum(1 for char in x if char in string.punctuation))
    X = emails['text']
    y = emails['spam']
    #Runs a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #Makes a pipeline for vectorizing(words to numbers) and then classifying(trains model)
    spam_model = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
    ])
    #Feed data through pipeline --- MODEL CREATED(Ran tests and got 99% accuracy)
    spam_model.fit(X_train, y_train)
    return spam_model

loaded_spam_model = model()

##CREATES DATAFRAME
def create_df(input):
    #GET DATA
    df = pdf.return_df(input)
    #SPAM
    nlp = spacy.load('en_core_web_md')
    df['spam'] = loaded_spam_model.predict(df['docs'])
    #GRADE DISPLAYER
    def find_assignment_grades(text):
        toReturn = []
        start_index = text.find("Assignment Graded:")
        end_index = text.find("@terpmail")
        if(start_index == -1 or end_index == -1):
            return '$$'
        s = text[start_index + len("Assignment Graded:"):end_index].strip()
        for part in s.split(', '):
            toReturn.append(part.strip()[:20] + '...')
        return toReturn
    for index, row in df.iterrows():
        df.at[index, 'assignment'] = find_assignment_grades(row['docs'])[0]
        df.at[index, 'course'] = find_assignment_grades(row['docs'])[1]
    #SENTIMENT ANALYSIS
    sid = SentimentIntensityAnalyzer()
    df['scores'] = df['docs'].apply(lambda review: sid.polarity_scores(review))
    df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])
    def get_sentiment_label(compound):
        if compound >= 0.75:
            return "very positive"
        elif compound >= 0.25:
            return "positive"
        elif compound > 0:
            return "slightly positive"
        elif compound == 0:
            return "neutral"
        elif compound > -0.25:
            return "slightly negative"
        elif compound > -0.75:
            return "negative"
        else:
            return "very negative"
    df['sentiment_label'] = df['compound'].apply(get_sentiment_label)
    df.drop(['scores'], axis=1, inplace=True)
    #KEY DATES
    def find_date_sent(text):
        end_index = text.find("M")
        if(end_index == -1):
            return '$'
        return text[:end_index] + 'M'
    for index, row in df.iterrows():
        df.at[index, 'date_sent'] = find_date_sent(row['docs'])
    #KEY ENTITY SEARCH
    df['norp'] = '$'
    df['money'] = "$"
    df['product'] = '$'
    df['gpe'] = '$'
    df['dates'] = '$'
    df['times'] = '$'
    for index, row in df.iterrows():
        doc = nlp(df.at[index,'docs'])
        norp_list = ''
        money_list = ''
        product_list = ''
        gpe_list = ''
        dates_list = ''
        times_list = ''
        for entity in doc.ents:
            if entity.label_ == "NORP":
                if norp_list.find(str(entity.text)) == -1 :
                    norp_list += entity.text + ", "
            elif entity.label_ == "MONEY":
                if money_list.find(str(entity.text)) == -1 :
                    money_list += entity.text + ", "
            elif entity.label_ == "PRODUCT":
                if product_list.find(str(entity.text)) == -1 :
                    product_list += entity.text + ", "
            elif entity.label_ == "GPE":
                if gpe_list.find(str(entity.text)) == -1 :
                    gpe_list += entity.text + ", "
            elif entity.label_ == "DATE":
                if dates_list.find(str(entity.text)) == -1 :
                    dates_list += entity.text + ", "
            elif entity.label_ == "TIME":
                if times_list.find(str(entity.text)) == -1 :
                    times_list += entity.text + ", "
        df.at[index,'norp'] = norp_list
        df.at[index,'money'] = money_list
        df.at[index,'product'] = product_list
        df.at[index,'gpe'] = gpe_list
        df.at[index,'dates'] = dates_list
        df.at[index,'times'] = times_list

        return df

def limit_words(text, num_words=6):
    words = text.split()
    if len(words) > num_words:
        words = words[:num_words]
        return ' '.join(words) + '...'
    else:
        return text
    

def display(type, df):
    df['preview'] = df['docs'].astype(str)
    df['preview'] = df['preview'].apply(lambda x: limit_words(x, 6))
    df['dates'] = df['dates'].apply(lambda x: limit_words(x, 2))
    df['times'] = df['times'].apply(lambda x: limit_words(x, 2))
    ham_df = df[df['spam']==0]
    if type == 'raw':
        df.drop(['docs'], axis = 1, inplace = True)
        return df
    elif type == 'ham':
        ham_df = ham_df[['preview','date_sent','sentiment_label']]
        return ham_df
    elif type == 'spam':
        spam_df = df[df['spam']==1]
        spam_df = spam_df[['preview','date_sent']]
        return spam_df
    elif type == 'dates':
        ham_df = ham_df[['preview','date_sent','sentiment_label']]
        return ham_df.sort_values(by=['date_sent'])
    elif type == 'graded':
        ham_df = ham_df[['preview','date_sent', 'assignment', 'course','dates','times']]
        return ham_df[ham_df['assignment']!='$']
    elif type == 'sentiment':
        ham_df = ham_df[['preview','date_sent','sentiment_label','compound']]
        return ham_df.sort_values(by=['compound'])
    else:
        return 'Not supported.'