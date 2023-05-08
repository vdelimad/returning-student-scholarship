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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering
import streamlit.components.v1 as stc
from wordcloud import WordCloud


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

# custom list of phrases
compound_phrases = ['data science', 'machine learning', 'artificial intelligence', 'neural network', 'deep learning', 'software engineering', 'computer science', 'team work', 'computer_vision', 'neural networks', 'reinforcement learning', 'web development', 'data tools', 'statistical analysis', 'written communication', 'data modelling', 'data modeling', 'time series', 'natural language processing', 'big data', 'data analyst', 'cloud computing', 'natural language', 'scikit learn', 'text data', 'information system', 'information systems', 'smart contract', 'verbal communication', 'problem solving', 'hands on', 'hand on', 'detail oriented', 'fast paced', 'web design', 'project management', 'front end', 'back end', 'ph d', 'natural language understanding', 'programming language', 'programming languages', 'consensus protocols', 'business solutions', 'distributed system', 'distributed systems', 'software development', 'web application', 'web applications', 'business intelligence', 'deep understanding', 'life insurance', 'base salary', 'base pay', 'tuition assistance', 'parental leave', 'paid leave', 'paid vacation', 'sick leave', 'benefits package', 'paid time off', 'stock options', 'paid vacation', 'paid holidays', 'health insurance', 'retirement benefits', 'salary benefits', 'disability insurance', 'competitive benefits', 'competitive salary', 'annual bonus','work life balance']

# Custom list of stopwords
custom_stopwords = ['experience', 'qualifications', 'ability', 'based', 'well', 'also', 'help', 'requirements', 'including', 'skills', 'related', 'required', 'field', 'using', 'knowledge', 'strong', 'etc', 'proficiency', 'e', 'excellent', 'relevant', 'g', 'least', 'years', 'must', 'work', 'demonstrated', 'one', 'two', 'similar', 'able', 'proven', 'working', 'team', 'developing', 'candidate', 'background', 'equivalent', 'applying', 'effectively', 'may', 'min', 'minimum', 'range', 'comprehensive', 'salary', 'eligible', 'k', 'role', 'include', 'per', 'position', 'company', 'long', 'us','use', 'employees']

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


# helper to render html
#def render_html(filepath):
#    with open(filepath, 'r') as f:
#        html = f.read()
#    stc.iframe(html, width=700, height=500)

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
    st.markdown("## Discovering Great Companies and Polishing Your Resume")
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
            
            

    st.markdown("A job search can be very intimidating, complex, and time-consuming. There are several aspects to consider, such as goals and motivations, document preparation, and networking skills for interviews and salary negotiations. In this project, we seek to aid our readers in optimizing their search for great companies and polishing their resumes to be strong competitors in the resume-ranking software era.")
    
    st.markdown("The importance of finding the right job cannot be understated. For students graduating, finding the right place to begin their careers can significantly shape the track or specialization to follow. For international students, getting a good, stable job can provide stability to meet visa sponsorship requirements. Even for experienced workers, finding the right place to work can translate into opportunities for a career change or even to get a chance to jump back into the field after a hiatus.")
    
    st.markdown("First, we start by performing a literature review on what field experts recommend when engaging in a job search. Then we dive into the companies offering data science and data science-related positions from our database of Google searches. Then, we provide a few tools and visualizations on optimizing the keywords in your résumé before finally diving into conclusions and future work.")
    
    
    # transitions        
    st.markdown("---")
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






def finding_companies_page():

    st.markdown("## Finding Companies")
    
    
    st.markdown("The natural place to begin is to find companies located in regions where you live or are willing to relocate. As an initial view, Figure 1 shows the cities in our data set where the companies are located, aggregated by state (for a detailed description of the data set, see the `The Data` section). Since our data set focused primarily on Washington, D.C. searches, this area has the most companies. However, D.C. aside, we can see California is the clear runner-up. Zooming into the plot shows that Illinois, New York, and Texas are the next contenders. Notably, states such as Utah, Denver, and Florida have few data science jobs.")
    
    
    
    # begin folium plot
    ######################################################################
    
    
    location_counts = jobs_results.dropna(subset=['latitude', 'longitude'])

    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

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
        st.markdown('#### Map of Job Locations Density')
        folium_static(m)
        st.caption('Figure 1: ')

        
    with folim_col3:
        st.write("")
        
    
    
    
    
    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>Tip: Click on a colorized cluster in the map to expand that region.</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    st.markdown("Next, in Table 1, we offer a job finder that can be filtered by state. The table shows the job title, company name, schedule type describing whether the job is full-time or otherwise, data science job category, and city location. Using California as an example, we can filter for the state and sort by company name. We see that Apple offers multiple jobs in this state and that the general diversity of employers is very vast. Similarly, the number of Reinforcement Learning and Natural Language Processing jobs stand out, given that these fields have increased in popularity in recent years. Potential job locations include the San Francisco Bay, San Jose, and Los Angeles.")

    
    
    # begin jobs table
    ######################################################################
    
    
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

    st.markdown('#### Job Table')

    # state selection
    state_selected = st.selectbox("Select a state:", unique_states)

    # get the dataframe for the selected state
    df = tables[state_selected]

    # display the dataframe as a table without row numbers (index)
    st.dataframe(df.reset_index(drop=True), use_container_width=True)
    
    
    st.caption('Table 1: ')




    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>Tip: Click on a column header to sort the table.</p>
        </div>
        """, unsafe_allow_html=True)
    
    
    
    
    st.markdown("Lastly, to further motivate finding the right employer, in Figure 2, we can see the most common benefits in our data science job data set. Data Science is a well-compensating field at the present time. Still, it’s also very competitive, and hence finding the right company to work for will impact how many of the benefits shown below a potential candidate will have access to through their compensation package.")



    
    # begin word cloud
    ######################################################################
    
    # Combine all descriptions into a single list of words
    all_words = []
    for desc in jobs_results['benefits']:
        words = preprocess_text(desc, compound_phrases, custom_stopwords, equivalent_phrases)
        all_words.extend(words)

    # Count word frequencies
    word_freq = Counter(all_words)

    # Generate the wordcloud from the top 50 words
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50])
    wc = WordCloud(background_color='white', max_words=50, width=800, height=400).generate_from_frequencies(top_words)


    # show
    folim_col1, folim_col2, folim_col3 = st.columns([1,6,1])
    
    with folim_col1:
        st.write("")
        
    with folim_col2:
        st.markdown('#### Job Benefits Text Word Cloud')
        st.image(wc.to_array(), use_column_width=True)
        st.caption('Figure 2: ')

        
    with folim_col3:
        st.write("")
        

        
    st.markdown("After finding good companies, we now turn to getting resumes polished in the next section.")

        






        
        
    
    
    
    
    
    
    
    
    
    # page transitions
    ######################################################################
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
    
    
    
    st.markdown("First, it is crucial to understand that the field is very competitive and that getting the details right can make a significant difference. Figure 3 shows a network depicting the cosine similarity between the count vectorized words in the `responsibilities` column for a subset of 50 randomly selected job postings from the data set (due to computational complexity). Given that all jobs are Data Science-related, we can see a central cloud stand out as the main shape of the network. Interestingly, however, Blockchain, Natural Language Processing, and Reinforcement Learning are more likely to be on the edges of the cloud, highlighting the skill sets from these jobs that differ from traditional Machine Learning jobs. Nonetheless, most job requirements are very similar, so it’s imperative to get the right keywords into the resume to maximize the chances of passing the resume ranking software filters.")


    
    
    # begin word network
    ######################################################################
    

    # Remove empty rows and reset index
    jobs_results_responsibilities = jobs_results.dropna(subset=['responsibilities'])
    jobs_results_responsibilities.reset_index(drop=True, inplace=True)


    # Select a random sample of 50 rows
    sample_size = 50
    jobs_results_responsibilities = jobs_results_responsibilities.sample(n=sample_size, random_state=42)
    jobs_results_responsibilities.reset_index(drop=True, inplace=True)


    # Define the stopwords to remove
    stop_words = set(stopwords.words('english'))

    # Create a CountVectorizer object
    vectorizer = CountVectorizer(stop_words=stop_words)

    # Count vectorize the 'description' column
    X = vectorizer.fit_transform(jobs_results_responsibilities['responsibilities'])

    # Convert the sparse matrix to a dense matrix
    X = X.toarray()

    # Calculate the cosine similarities between the job listings
    cos_sim = cosine_similarity(X)

    # Perform Agglomerative Clustering on the cosine similarity matrix
    n_clusters = 20
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    cluster_labels = cluster_model.fit_predict(1 - cos_sim)

    # Create a graph object
    G = nx.Graph()

    # Add nodes to the graph
    for i in range(len(jobs_results_responsibilities)):
        G.add_node(i, label=jobs_results_responsibilities.loc[i, 'title'], company=jobs_results_responsibilities.loc[i, 'company_name'])

    # Add edges to the graph
    for i in range(len(jobs_results_responsibilities)):
        for j in range(i+1, len(jobs_results_responsibilities)):
            G.add_edge(i, j, weight=cos_sim[i][j])

    # Calculate the positions of the nodes using the spring layout
    pos = nx.spring_layout(G, k=1, seed=42)

    # Extract the X and Y coordinates of the nodes
    x_coords = [pos[i][0] for i in range(len(pos))]
    y_coords = [pos[i][1] for i in range(len(pos))]

    # Create a Plotly scatter plot
    fig = go.Figure()

    # Add lines for edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(width=1, color='gray'), showlegend=False, hoverinfo='none'))

    # 

    # Add the scatter plot trace
    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers',
                            marker=dict(size=20, color=cluster_labels, colorscale='Viridis', showscale=False),
                            text=[f"{data['label']}<br>{data['company']}" for i, data in G.nodes(data=True)],
                            hoverinfo='text', showlegend=False))





    # Update the layout of the plot
    fig.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    hovermode='closest',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=30, b=0))

    
    # show
    network_viz_col1, network_viz_col2, network_viz_col3 = st.columns([1,6,1])
    
    with network_viz_col1:
        st.write("")
        
    with network_viz_col2:
        st.markdown('#### Similarities in Job Responsibilities Detail')
        st.plotly_chart(fig)
        st.caption('Figure 3: ')

        
    with network_viz_col3:
        st.write("")
    
    
    
    
    
    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>Tip: Hover over the nodes to the the job title and company</p>
        </div>
        """, unsafe_allow_html=True)

    
    



    
    st.markdown("Next, in Figure 4 below, we can see a tag cloud showing the most frequent words in the `qualifications` columns of our data set, with a dropdown to filter for each job category. Depending on the job that interests the reader, these are essential keywords to include in a base resume. Incorporating them as a default will save a significant time when tailoring a resume for a specific job in the discipline, as many of the keywords will already have been strategically placed.")

    
    
    
    
    
    
    
    
    
    # begin tag cloud
    ######################################################################
    
    st.markdown('#### Job Qualifications Text Tag Cloud')
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


    # show
    
    tag_cloud_html = generate_tag_cloud_html(word_freq)
    st.markdown(tag_cloud_html, unsafe_allow_html=True)
    
    st.caption('Figure 4: ')


    st.markdown("Figure 4 shows several popular requirements for all categories, such as Python, C, and algorithms. Conversely, we can also see specialist skills such as TensorFlow and PyTorch in the Machine Learning category and Smart Contracts and Ethereum in the Blockchain category.")
    
    st.markdown("Next, to assist the reader, we offer the tool below in which a resume can be provided, and it will check if the keywords as per the selected category are present. To use the tool, a user can simply paste the resume into the textbox and click the `Check Missing Keywords` button.")

    
    

    # begin missing words check
    ######################################################################
    
    st.markdown('#### Resume Keywords Check Tool')

    # Input text
    input_text = st.text_area("Enter your resume text:")

    # Use the list of 50 words from the tag cloud
    word_list = [word for word, _ in word_freq]

    # Check for missing words
    if st.button("Check Missing Keywords"):
        if input_text and word_list:
            missing_words = check_missing_words(input_text, word_list)
            if missing_words:
                st.markdown(f"<p style='color: grey;'>The following words are missing from the text: {', '.join(missing_words)}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: grey;'>All words are present in the text.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: grey;'>Please enter the text to check.</p>", unsafe_allow_html=True)

            
    
    
    st.markdown("Now that we have covered these critical aspects of a job search, we can move on to some concluding remarks.")

    
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



def the_data_page():

    st.markdown("## The Data")

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

if st.sidebar.button("The Data"):
    st.session_state.selected_page = "The Data"

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
elif st.session_state.selected_page == "The Data":
    the_data_page()
    
elif st.session_state.selected_page == "The App":
    the_app_page()
    
elif st.session_state.selected_page == "References":
    references_page()

