
############### Import Libraries ###############
import streamlit as st
import pandas as pd

from textblob import TextBlob

## Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


page_bg_img = '''
<style>
.css-1v3fvcr {
background-image: url("https://nlpcloud.io/assets/images/dark-background.jpg");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

############### Main Page ###############

## Define Dataset
df = pd.read_csv("cleaned_data.csv")


#Sentiment Analysis Page
def SA_page():
    
    #Skip line
    st.write("""""") 
    
st.markdown("""
            <style> 
            h1{
                color: white;
                } 

            .css-8znj5f{
                color: black;
                } 
            p, ol, ul, dl{
                color: white;
                }

            .css-1e5imcs{
                color: white;
                }
            .css-1ekf893{
                color: white;
                font-weight: bold;
                font-size:  16px;
                }
            .css-qbe2hs{
                color: black;
                font-weight: bold;
                }
            .css-glyadz{
                color: white;
                font-size: 24px;
                }
            .code{
                color: black;
                }
            .st-fm{
                background-color:#29f16cad;
                }
            .css-10trblm{
                font-weight: bold;
                color: white;
                }
            hr{
                border-bottom: 2px solid white;
                }
            code{
                font-weight: bolder;
                }
            .css-h9oeas{
                color: white;
                }
            .css-120qjcf, .css-14n4bfl:hover{
                color: black;
                }
            .st-cy{
                background-color: rgb(9 171 59 / 35%);
                }
            .st-cq, .st-cp{
                background-color: rgb(234 16 16 / 45%);
                }            
            .css-gh49vm{
                background-color: lightgray;
                }
            
                                                </style>"""
            , unsafe_allow_html=True)    


#Title
st.title("""Sentiment Analysis Web Application
             
    This web application has been developed with model deployment. 
             
        """)

st.write('---')

## Input reviews
Sentence = st.text_area('Enter your reviews: ')
## Analyze Button
Analyzebtn = st.button('Analyze')

st.write('Click "Analyze" to show the results performed by Sentiment Analysis!')


## Get review from textbox to perform SA & predict
if Analyzebtn:
    
    ## Print raw review
    st.write('Reviews: ')
    #st.write(Sentence)
    st.success(Sentence)
    
    st.header("Subjectivity & Polarity")
    st.subheader("This section will analyze the polarity & subjectivity of the review.")
    
    ## Shows Subjectivity & Polarity (include Sentiment)
    st.write('Subjectivity & Polarity of the review: ')
    Subjectivity = TextBlob(Sentence).sentiment.subjectivity
    Polarity = TextBlob(Sentence).sentiment.polarity
    st.write("Subjectivity: ", Subjectivity)      
    st.write("Polarity: ", Polarity)
    
    
    ## Logistic Regression Model
    st.write('---')
    
    st.header("Predict Sentiment")
    st.subheader("This section will classify the sentiment of the review.")
    
    x_train, x_test, y_train, y_test = train_test_split(df['What is your opinion for Online Learning Implementation during COVID-19?'], 
                                                        df['Sentiment'], test_size = 0.2, random_state = 42)
    
    ## Define Vectorizer
    vectorizer = CountVectorizer(binary = True, stop_words = 'english')
    
    ## learn a vocabulary dictionary of all tokens in raw documents
    vectorizer.fit(list(x_train) + list(x_test))

    ## transform document into document-term matrix
    x_train_vectorizer = vectorizer.transform(x_train.values)
    x_test_vectorizer = vectorizer.transform(x_test.values)
    
    ## Oversampling
    sm = SMOTE(random_state=42)
    x_resampling, y_resampling = sm.fit_resample(x_train_vectorizer, y_train)
    
    ## Logistic Parameter Tuning
    # define hyperparameters and grid search 
    grid = dict(solver = ['newton-cg', 'lbfgs', 'liblinear', 'saga', 'sag'], C = [1, 10, 100, 1000])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    Optimize_Logistic = GridSearchCV(LogisticRegression(), param_grid = grid, 
                               n_jobs= -1, cv = cv, scoring = 'accuracy', error_score=0)
    
    ## Fit Optimize Logistic
    Logistic_grid_result = Optimize_Logistic.fit(x_resampling, y_resampling)
    
    ## Prediction
    if (Polarity == 0): ## The review will become neutral if polarity is 0
        st.write('This review is classified as')
        st.success('Neutral')
        
    else: 
        vectorize = vectorizer.transform([Sentence])
        result = Optimize_Logistic.predict(vectorize)
        result = result.tolist() ## Dataframe to list
        
        st.write('This review is classified as')
            
        if(result[0] == "Positive"):
            st.success(result[0])
        else:
            st.error(result[0])
    
    
else: 
    st.write("No text input!!!")


#Skip line
st.write("""""") 









    
    