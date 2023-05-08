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
from PIL import Image
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static


################################################
# data
################################################

jobs_results = pd.read_csv("Data/jobs_results_with_coords.csv")
# clean the state column
jobs_results['state'] = jobs_results['location'].apply(lambda x: x.split(', ')[-1]) # put in a separate column
jobs_results['state'] = jobs_results['state'].apply(lambda x: x.split('(')[0].strip()) # remove the parenthesis and trim the whitespace


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

    # fix smart contracts
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
             <p>All code used in this project is publicly available on <a href="https://github.com/vdelimad/returning-student-scholarship" target="_blank">GitHub</a>. <br>
            <mark>Warning!</mark> This GitHub link may contain student identifiers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## Introduction")

    
    # ref: https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        image = Image.open('Code/Streamlit/Images/pexels-anna-tarazevich-5598295.jpg')
        st.image(image, caption='Photo credit: Tarazevich (2020)', width=600)

    with col3:
        st.write("")
            
            
    # transitions        
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.write("")   
    
    with col2:
        st.write("")
           
    with col3:
        if st.button(":arrow_right: The Literature"):
            st.session_state.selected_page = "The Literature"
            st.experimental_rerun()

        
        

    



def the_literature_page():

    st.markdown("## The Literature")
    
    st.markdown("Before considering the specific actions to take when performing a job search, getting into the right mindset is essential. Rothman (2014) compares the job search process with managing a project. As your own project manager, it is crucial to strategize and think about what is the most efficient use of time. For instance, Rothman emphasizes the importance of collecting notes about the process to better understand your working style and rhythm. Not only does this serve as a source of personal feedback, but also, a significant amount of research shows that keeping track of your accomplishments will support your motivation and self-esteem (Amabile 2014).")

    st.markdown("It is also essential to focus on one action at a time. A job search can be daunting and very time-consuming; deciding how much work to focus on every week can be the key to success. For example, techniques such as developing a personal Kanban board and keeping tasks within one-week time boxes can also be very effective organizational methods and help avoid multitasking.")

    st.markdown("Once the organizational style is planned, we can consider the elements needed for the job search. Ceniza-Levine and Thanasoulis-Cerrachio (2011) discuss the importance of evaluating your specific life situation before defining a strategy since these factors will significantly influence the job search preparation required for success. For example, a student looking for an internship and a student about to graduate have very different goals. Graduating students may prioritize financial self-sufficiency, while ongoing students may instead focus on acquiring valuable experience. Similarly, experienced candidates may want to change industries or return to the workforce after a hiatus. These considerations come with their own deadlines, access to resources, and emotional constraints.")

    st.markdown("Once these considerations are considered, Ceniza-Levine and Thanasoulis-Cerrachio suggest breaking down the process into six sequential steps. Step one is all about making up your mind on which career track you wish to pursue. Steps two and three involve preparing documents and performing research to guide and support the application process. Finally, steps four to six involve the process of networking, interviewing, and moving forward all the way until closing an offer. In this project, we assume our readers have already chosen to pursue a subfield within data science and focus on helping them with steps two and three: finding companies and preparing a resume, which we discuss next.")

    st.markdown("When browsing for potential employers, Dalton (2012) suggests narrowing down the surge to approximately 40 companies. This will allow each applicant to focus more on tailoring the profile to a reasonable number of options since the search space is extensive. For instance, the Small Business & Entrepreneurship Council (2023) reported 6.1 million employer firms in the U.S. as of 2019 using data from U.S. Census Bureau. In addition, there are methods such as LAMP (list, alumni, motivation, posting) which are very detailed in their approach, although just the conscious search for feasible options may suffice.")

    st.markdown("For résumé building, many companies use software that ranks resumes based on their relevance to the job description. To do this, they check the résumé for particular keywords and score the résumé accordingly. Furthermore, even when the resume is not being scored by software, recruiters spend an average of six seconds on each résumé, which implies that they skim for keywords rather than perform a thorough read of the résumé (Resume Worded, 2023). These facts underscore the importance of getting the correct keywords into the resume.")

    st.markdown("In this project, we build on the recommendations discussed, focusing on company search and résumé preparation to analyze a job posting data set constructed with Google searches.")



    # transitions
    st.markdown("---")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button(":arrow_left: Introduction"):
            st.session_state.selected_page = "Introduction"
            st.experimental_rerun()    
    
    with col2:
        st.write("")
           
    with col3:
        if st.button(":arrow_right: Finding Companies"):
            st.session_state.selected_page = "Finding Companies"
            st.experimental_rerun()


import streamlit.components.v1 as stc

def render_html(filepath):
    with open(filepath, 'r') as f:
        html = f.read()
    stc.iframe(html, width=700, height=500)



def finding_companies_page():

    st.markdown("## Finding Companies")
    
    location_counts = jobs_results.dropna(subset=['latitude', 'longitude'])
    
    


    m = folium.Map(location=[location_counts['latitude'].mean(), location_counts['longitude'].mean()], zoom_start=4)

    marker_cluster = MarkerCluster().add_to(m)

    for idx, row in location_counts.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']], 
            tooltip=row['location']
        ).add_to(marker_cluster)

    folim_col1, folim_col2, folim_col3 = st.columns([1,6,1])
    
    with folim_col1:
        st.write("")
        
    with folim_col2:
        folium_static(m)
        
    with folim_col3:
        st.write("")
    
    
    
    
    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>Tip: Click on a colorized cluster in the map to expand that region.</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    
    

    # get unique states
    unique_states = sorted(jobs_results['state'].unique())

    # move 'Anywhere' first
    if 'Anywhere' in unique_states:
        unique_states.remove('Anywhere')
        unique_states.insert(0, 'Anywhere')

    # create a table for each state
    tables = {}
    for state in unique_states:
        filtered_jobs_results = jobs_results[jobs_results['state'] == state]
        filtered_jobs_results = filtered_jobs_results[['title', 'company_name', 'schedule_type', 'category', 'location']]
        
        # capitalize headers and replace underscores with spaces
        filtered_jobs_results.columns = [col.replace('_', ' ').title() for col in filtered_jobs_results.columns]
        tables[state] = filtered_jobs_results

    # Streamlit app
    st.title("Job finder")

    # state selection
    state_selected = st.selectbox("Select a state:", unique_states)

    # get the dataframe for the selected state
    df = tables[state_selected]

    # display the dataframe as a table without row numbers (index)
    st.dataframe(df.reset_index(drop=True), use_container_width=True)



    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>Tip: Click on a column header to sort the table.</p>
        </div>
        """, unsafe_allow_html=True)




    
    
    
    
    
    
    
    # transitions
    st.markdown("---")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button(":arrow_left: The Literature"):
            st.session_state.selected_page = "The Literature"
            st.experimental_rerun()
    
    with col2:
        st.write("")
           
    with col3:
        if st.button(":arrow_right: Resume Keywords"):
            st.session_state.selected_page = "Resume Keywords"
            st.experimental_rerun()



def resume_keywords_page(jobs_results, compound_phrases, custom_stopwords, row_length=70):
    
    st.markdown('## Resume Keywords')

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



    st.markdown("### Check your resume")

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
            
    
    # transitions
    st.markdown("---")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button(":arrow_left: Finding Companies"):
            st.session_state.selected_page = "Finding Companies"
            st.experimental_rerun()
    
    with col2:
        st.write("")
           
    with col3:
        if st.button(":arrow_right: Conclusions"):
            st.session_state.selected_page = "Conclusions"
            st.experimental_rerun()






def conclusions_page():

    st.markdown("## Conclusions")


    # transitions        
    st.markdown("---")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button(":arrow_left: Resume Keywords"):
            st.session_state.selected_page = "Resume Keywords"
            st.experimental_rerun()
    
    with col2:
        st.write("")
           
    with col3:
        st.write("")



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
    
if st.sidebar.button("Finding Companies"):
    st.session_state.selected_page = "Finding Companies"
        
if st.sidebar.button("Resume Keywords"):
    st.session_state.selected_page = "Resume Keywords"
    
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

elif st.session_state.selected_page == "Finding Companies":
    finding_companies_page()

elif st.session_state.selected_page == "Resume Keywords":
    resume_keywords_page(jobs_results, compound_phrases, custom_stopwords)

elif st.session_state.selected_page == "Conclusions":
    conclusions_page()

# about section
elif st.session_state.selected_page == "Data":
    data_page()
elif st.session_state.selected_page == "The App":
    the_app_page()
elif st.session_state.selected_page == "References":
    references_page()

