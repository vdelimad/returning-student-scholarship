################################################
# import libraries
################################################


import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re
import matplotlib.pyplot as plt

################################################
# data
################################################

jobs_results = pd.read_csv("Data/jobs_results_with_coords.csv")



################################################
# vars and lists
################################################

# Custom list of compound phrases
compound_phrases = ['data science', 'machine learning', 'artificial intelligence', 'neural network', 'deep learning', 'software engineering', 'computer science', 'team work', 'computer_vision', 'neural networks', 'reinforcement learning', 'web development', 'data tools', 'statistical analysis', 'written communication', 'data modelling', 'data modeling', 'time series', 'natural language processing', 'big data', 'data analyst', 'cloud computing', 'natural language', 'scikit learn', 'text data', 'information system', 'information systems', 'smart contract', 'verbal communication', 'problem solving', 'hands on', 'hand on', 'detail oriented', 'fast paced', 'web design', 'project management', 'front end', 'back end', 'ph d', 'natural language understanding', 'programming language', 'programming languages', 'consensus protocols', 'business solutions', 'distributed system', 'distributed systems', 'software development', 'web application', 'web applications', 'business intelligence', 'deep understanding']

# Custom list of stopwords
custom_stopwords = ['experience', 'qualifications', 'ability', 'based', 'well', 'also', 'help', 'requirements', 'including', 'skills', 'related', 'required', 'field', 'using', 'knowledge', 'strong', 'etc', 'proficiency', 'e', 'excellent', 'relevant', 'g', 'least', 'years', 'must', 'work', 'demonstrated', 'one', 'two', 'similar', 'able', 'proven', 'working', 'team', 'developing', 'candidate', 'background', 'equivalent', 'applying', 'effectively', 'may', 'min', 'minimum']

equivalent_phrases = {
    'machine_learning': ['ml', 'machine learning'],
    'artificial_intelligence': ['ai', 'artificial_intelligence'],
    'natural_language_processing': ['nlp', 'natural language processing'],
    'natural_language_understanding': ['nlu', 'natural language understanding'],
    'business intelligence': ['bi', 'business intelligence'],
    'phd': ['phd', 'ph d'],

}



################################################
# helper functions
################################################


# tag cloud functions
def replace_compound_phrases(text, phrases):
    for phrase in phrases:
        token = phrase.replace(' ', '_')
        pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
        text = pattern.sub(token, text)
    return text

def preprocess_text(text, compound_phrases, custom_stopwords, equivalent_phrases):
    if text is None or isinstance(text, float):  # check if input is None or a float
        return []

    # Convert to string and remove leading/trailing white space
    text = str(text).strip()

    # Replace 'smart contract' and 'smart contracts' with 'smart_contract'
    text = re.sub(r'\bsmart contracts?\b', 'smart_contract', text, flags=re.IGNORECASE)

    # Remove special characters and digits
    text = re.sub(r'\W|\d', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Tokenize the text
    words = nltk.word_tokenize(text)

    # Replace equivalent phrases
    replaced_words = []
    for word in words:
        replaced = False
        for key, phrases in equivalent_phrases.items():
            if word in phrases:
                replaced_words.append(key)
                replaced = True
                break
        if not replaced:
            replaced_words.append(word)

    # Replace compound phrases with a single token
    text = ' '.join(replaced_words)
    text = replace_compound_phrases(text, compound_phrases)
    words = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english')).union(set(custom_stopwords))
    words = [word for word in words if word not in stop_words]

    return words



def split_words_into_rows(words, row_length):
    rows = []
    current_row = []
    current_length = 0

    for word, count in words:
        if current_length + len(word) + len(str(count)) + 2 > row_length:
            rows.append(current_row)
            current_row = []
            current_length = 0

        current_row.append((word, count))
        current_length += len(word) + len(str(count)) + 2

    if current_row:
        rows.append(current_row)

    return rows

def create_visualization(words, row_length):
    fig, ax = plt.subplots(figsize=(12, 8))

    rows = split_words_into_rows(words, row_length)

    y = 0
    for row in rows:
        x = 0
        for word, count in row:
            ax.text(x, y, word, fontsize=12, color='black')
            x += len(word) * 15  # Increase the scaling factor if necessary
            ax.text(x, y, f"({count})", fontsize=12, color='gray')
            x += len(str(count)) * 15 + 15  # Add additional space between words
        y -= 1.5 * 20   # Increase the scaling factor for row spacing

    ax.axis('off')
    ax.set_xlim(0, 12 * 72)
    ax.set_ylim(-8 * 72, 0)
    return fig


def generate_tag_cloud_html(words):
    tag_cloud_html = "<div class='tag-cloud'>"

    for word, count in words:
        tag_cloud_html += f"<span class='tag-word'>{word} <span class='tag-count'>({count})</span></span> "

    tag_cloud_html += "</div>"

    return tag_cloud_html




# missing words functions
def check_missing_words(input_text, word_list):
    missing_words = []
    input_words = set(input_text.lower().split())

    for word in word_list:
        if word.lower() not in input_words:
            missing_words.append(word)

    return missing_words










################################################
# pages
################################################


def introduction_page():
    st.markdown("# The Data Science Job Search Survival Guide")
    st.markdown("## What to Expect and How to Maximize Your Chances")
    st.markdown("PUBLISHED: May 8, 2023")
    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>All code used in this project is publicly available on <a href="https://www.freecodecamp.org/" target="_blank">GitHub</a>. <br>
            <mark>Warning!</mark> This GitHub link may contain student identifiers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## Introduction")




def the_literature_page():

    st.markdown("## The Literature")




def tag_cloud_page(jobs_results, compound_phrases, custom_stopwords, row_length=70):
    st.title('Tag Cloud Visualization')

    # Get the category
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = jobs_results["category"].iloc[0]

    categories = sorted(jobs_results["category"].unique())
    category = st.selectbox("Select a category:", categories, index=categories.index(st.session_state.selected_category), key='selected_category_widget')
    if category != st.session_state.selected_category:
        st.session_state.selected_category = category
        st.experimental_rerun()

    st.session_state.selected_category = category

    # Filter data based on the selected category
    filtered_data = jobs_results[jobs_results["category"] == st.session_state.selected_category]

    # Process the text
    all_words = []
    for desc in filtered_data['qualifications']:
        words = preprocess_text(desc, compound_phrases, custom_stopwords, equivalent_phrases)
        all_words.extend(words)

    # Count word frequencies
    word_freq = Counter(all_words).most_common(50)

    # Create the visualization
    #fig = create_visualization(word_freq, row_length)
    #st.pyplot(fig)
    tag_cloud_html = generate_tag_cloud_html(word_freq)
    st.markdown(tag_cloud_html, unsafe_allow_html=True)








def word_checker_page():
    st.title("Missing Word Checker")

    # Input text
    input_text = st.text_area("Enter your large amount of text:")

    # List of words to check
    word_list_str = st.text_input("Enter the list of words to check (separated by commas):")
    word_list = [word.strip() for word in word_list_str.split(',') if word.strip()]

    # Check for missing words
    if st.button("Check Missing Words"):
        if input_text and word_list:
            missing_words = check_missing_words(input_text, word_list)
            if missing_words:
                st.write(f"The following words are missing from the text: {', '.join(missing_words)}")
            else:
                st.write("All words are present in the text.")
        else:
            st.write("Please enter both the text and the list of words to check.")





def conclusions_page():

    st.markdown("## Conclusions")




def data_page():

    st.markdown("## Data")

def the_app_page():

    st.markdown("## The App")


def references_page():

    st.markdown("## References")



################################################
# streamlit setup
################################################


# init streamlit
st.set_page_config(page_title="The Data Science Job Search Survival Guide", layout="wide")
button_clicked = False

# init default page
default_page = "Introduction"
page = default_page

# add custom css
with open('Code/Streamlit/custom.css') as f:
    st.markdown (f'<style>{f.read()}</style>', unsafe_allow_html=True)



# remove built in sidebar options (https://stackoverflow.com/questions/72543675/is-there-a-way-to-remove-the-side-navigation-bar-containing-file-names-in-the-la)
#no_sidebar_style = """
#    <style>
#        div[data-testid="stSidebarNav"] {display: none;}
#    </style>
#"""
#st.markdown(no_sidebar_style, unsafe_allow_html=True)




################################################
# view management
################################################

# selecting pages
##################################################

# main section
st.sidebar.subheader("Main")
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = default_page

if st.sidebar.button("Introduction"):
    st.session_state.selected_page = "Introduction"

if st.sidebar.button("The Literature"):
    st.session_state.selected_page = "The Literature"
    
if st.sidebar.button("Tag Cloud"):
    st.session_state.selected_page = "Tag Cloud"

if st.sidebar.button("Word Checker"):
    st.session_state.selected_page = "Word Checker"
    
if st.sidebar.button("Conclusions"):
    st.session_state.selected_page = "Conclusions"

# about section
st.sidebar.subheader("About")
if st.sidebar.button("Data"):
    st.session_state.selected_page = "Data"

if st.sidebar.button("The App"):
    st.session_state.selected_page = "The App"

if st.sidebar.button("References"):
    st.session_state.selected_page = "References"


# render pages
##################################################

# main section
if st.session_state.selected_page == "Introduction":
    introduction_page()
    
elif st.session_state.selected_page == "The Literature":
    the_literature_page()

elif st.session_state.selected_page == "Tag Cloud":
    tag_cloud_page(jobs_results, compound_phrases, custom_stopwords)
elif st.session_state.selected_page == "Word Checker":
    word_checker_page()

elif st.session_state.selected_page == "Conclusions":
    conclusions_page()

# about section
elif st.session_state.selected_page == "Data":
    data_page()
elif st.session_state.selected_page == "The App":
    the_app_page()
elif st.session_state.selected_page == "References":
    references_page()

