################################################
# import libraries
################################################

import streamlit as st
import pandas as pd
import nltk
#nltk.download()
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
#nltk.download('popular')
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
import streamlit.components.v1 as components



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
custom_list_of_phrases = ['data science', 'machine learning', 'artificial intelligence', 'neural network', 'deep learning', 'software engineering', 'computer science', 'team work', 'computer_vision', 'neural networks', 'reinforcement learning', 'web development', 'data tools', 'statistical analysis', 'written communication', 'data modelling', 'data modeling', 'time series', 'natural language processing', 'big data', 'data analyst', 'cloud computing', 'natural language', 'scikit learn', 'text data', 'information system', 'information systems', 'smart contract', 'verbal communication', 'problem solving', 'hands on', 'hand on', 'detail oriented', 'fast paced', 'web design', 'project management', 'front end', 'back end', 'ph d', 'natural language understanding', 'programming language', 'programming languages', 'consensus protocols', 'business solutions', 'distributed system', 'distributed systems', 'software development', 'web application', 'web applications', 'business intelligence', 'deep understanding', 'life insurance', 'base salary', 'base pay', 'tuition assistance', 'parental leave', 'paid leave', 'paid vacation', 'sick leave', 'benefits package', 'paid time off', 'stock options', 'paid vacation', 'paid holidays', 'health insurance', 'retirement benefits', 'salary benefits', 'disability insurance', 'competitive benefits', 'competitive salary', 'annual bonus','work life balance']

# custom list of stopwords
custom_list_of_stopwords = ['experience', 'qualifications', 'ability', 'based', 'well', 'also', 'help', 'requirements', 'including', 'skills', 'related', 'required', 'field', 'using', 'knowledge', 'strong', 'etc', 'proficiency', 'e', 'excellent', 'relevant', 'g', 'least', 'years', 'must', 'work', 'demonstrated', 'one', 'two', 'similar', 'able', 'proven', 'working', 'team', 'developing', 'candidate', 'background', 'equivalent', 'applying', 'effectively', 'may', 'min', 'minimum', 'range', 'comprehensive', 'salary', 'eligible', 'k', 'role', 'include', 'per', 'position', 'company', 'long', 'us','use', 'employees']

# custom list of equivalent phrases
custom_list_of_equivalent_phrases = {
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


# function based on https://discuss.streamlit.io/t/no-way-to-set-focus-at-top-of-screen-on-page-reload-really/15474/14
def scroll_to_top_with_counter():
    if "counter" not in st.session_state:
        st.session_state.counter = 1

    st.session_state.counter += 1

    components.html(
        f"""
            <p>{st.session_state.counter}</p>
            <script>
                window.parent.document.querySelector('section.main').scrollTo(0, 0);
            </script>
        """,
        height=0
    )



# tag cloud functions
def replace_compound_phrases(text, phrases):
    for phrase in phrases:
        token = phrase.replace(' ', '_')
        pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
        text = pattern.sub(token, text)
    return text

def preprocess_text(text, compound_phrases, custom_stopwords, equivalent_phrases):
    
    if text is None or isinstance(text, float):  # check empty of float
        return []

    # remove white space
    text = str(text).strip()

    # fix smart contracts
    text = re.sub(r'\bsmart contracts?\b', 'smart_contract', text, flags=re.IGNORECASE)

    # remove digits / special characters
    text = re.sub(r'\W|\d', ' ', text)

    # change to lowercase
    text = text.lower()

    # tokenize with nltk
    words = nltk.word_tokenize(text)

    # fix equivalent phrases
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



    # replace compound phrases
    text = ' '.join(replaced_words)
    text = replace_compound_phrases(text, compound_phrases)
    words = nltk.word_tokenize(text)

    # remove stopwords
    stop_words = set(stopwords.words('english')).union(set(custom_stopwords))
    words = [word for word in words if word not in stop_words]

    return words



def split_words_into_rows(words, row_length):

    rows = []
    current_row = []
    current_length = 0

    # check if length of words, count and spaces is larger than row
    for word, count in words:
        if current_length + len(word) + len(str(count)) + 2 > row_length:

            # append if true and reinitiate list
            rows.append(current_row)
            current_row = []
            current_length = 0

        current_row.append((word, count))
        current_length += len(word) + len(str(count)) + 2

    if current_row:
        rows.append(current_row)

    return rows


def generate_tag_cloud_html(words):
    tag_cloud_html = "<div class='tag-cloud'>"

    # create the tag cloud
    for word, count in words:
        tag_cloud_html += f"<span class='tag-word'>{word} <span class='tag-count'>({count})</span></span> "

    tag_cloud_html += "</div>"

    return tag_cloud_html




# missing words functions
def check_missing_words(input_text, word_list):
    missing_words = []
    input_words = set(re.findall(r'\b\w+\b', input_text.lower())) # to make sure commas don't break the comparison

    # store words that are not present
    for word in word_list:
        if word.lower() not in input_words:
            missing_words.append(word)

    return missing_words





################################################
# pages
################################################


def introduction_page():
    
    scroll_to_top_with_counter()
    st.markdown("# The Data Science Job Search Survival Guide")
    st.markdown("## Discovering Great Companies and Polishing Your Resume")
    st.markdown("AUTHOR: Victor De Lima")
    st.markdown("PUBLISHED: May 8, 2023")
    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
        <ul>
            <li>All code used in this project is publicly available on <a href="https://github.com/vdelimad/returning-student-scholarship" target="_blank">GitHub</a>.</li>
            <li>To explore more of my work, please visit my <a href="https://victordelima.com/" target="_blank">website</a>.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Introduction")



    # ref: https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        image = Image.open('Code/Streamlit/Images/pexels-markus-winkler-4101343.jpg')
        st.image(image, caption='Photo credit: Winkle (2020)', width=500)

    with col3:
        st.write("")



    st.markdown("A job search can be very intimidating, complex, and time-consuming. There are several aspects to consider, such as goals and motivations, document preparation, and networking skills for interviews and salary negotiations. In this project, we seek to aid our readers in optimizing their search for great companies and polishing their resumes to be strong competitors in the resume-ranking software era.")

    st.markdown("The importance of finding the right job cannot be understated. For students graduating, finding the right place to begin their careers can significantly shape the track or specialization to follow. For international students, getting a good job can provide stability to meet visa sponsorship requirements. Even for experienced workers, finding the right place to work can translate into opportunities for a career change or even to get a chance to jump back into the field after a hiatus.")

    st.markdown("First, we start by performing a literature review on what field experts recommend when engaging in a job search. Then we dive into the companies offering Data Science and Data Science-related positions from our database of Google searches. Then, we provide a few tools and visualizations on optimizing the keywords in a resume before finally diving into conclusions and future work.")


    # page transitions
    ######################################################################      
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
    
    scroll_to_top_with_counter()
    
    st.markdown("Before considering the specific actions to take when performing a job search, getting into the right mindset is essential. Rothman (2014) compares the job search process with managing a project. As your own project manager, it is crucial to strategize and think about what is the most efficient use of time. For instance, Rothman emphasizes the importance of collecting notes about the process to better understand your working style and rhythm. Not only does this serve as a source of personal feedback, but also, a significant amount of research shows that keeping track of your accomplishments will support your motivation and self-esteem (Amabile 2014).")

    st.markdown("It is also essential to focus on one action at a time. A job search can be daunting and very time-consuming; deciding how much work to focus on every week can be the key to success. For example, techniques such as developing a personal Kanban board and keeping tasks within one-week time boxes can also be very effective organizational methods and help avoid multitasking.")

    st.markdown("Once the organizational style is planned, we can consider the elements needed for the job search. Ceniza-Levine and Thanasoulis-Cerrachio (2011) discuss the importance of evaluating your specific life situation before defining a strategy since these factors will significantly influence the job search preparation required for success. For example, a student looking for an internship and a student about to graduate have very different goals. Graduating students may prioritize financial self-sufficiency, while ongoing students may instead focus on acquiring valuable experience. Similarly, experienced candidates may want to change industries or return to the workforce after a hiatus. These considerations come with their own deadlines, access to resources, and emotional constraints.")

    st.markdown("Once these considerations are considered, Ceniza-Levine and Thanasoulis-Cerrachio suggest breaking down the process into six sequential steps. Step one is all about making up your mind on which career track you wish to pursue. Steps two and three involve preparing documents and performing research to guide and support the application process. Finally, steps four to six involve the process of networking, interviewing, and moving forward all the way until closing an offer. In this project, we assume our readers have already chosen to pursue a subfield within data science and focus on helping them with steps two and three: finding companies and preparing a resume, which we discuss next.")

    st.markdown("When browsing for potential employers, Dalton (2012) suggests narrowing down the surge to approximately 40 companies. This will allow each applicant to focus more on tailoring the profile to a reasonable number of options since the search space is extensive. For instance, the Small Business & Entrepreneurship Council (2023) reported 6.1 million employer firms in the U.S. as of 2019 using data from U.S. Census Bureau. In addition, there are methods such as LAMP (list, alumni, motivation, posting) which are very detailed in their approach, although just the conscious search for feasible options may suffice.")

    st.markdown("For resume building, many companies use software that ranks resumes based on their relevance to the job description. To do this, they check the resume for particular keywords and score the resume accordingly. Furthermore, even when the resume is not being scored by software, recruiters spend an average of six seconds on each resume, which implies that they skim for keywords rather than perform a thorough read of the resume (Resume Worded, 2023). These facts underscore the importance of getting the correct keywords into the resume.")

    st.markdown("In this project, we build on the recommendations discussed, focusing on company search and resume preparation to analyze a job posting data set constructed with Google searches.")



    # page transitions
    ######################################################################
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
    
    if "finding_companies_loaded" not in st.session_state:
        scroll_to_top_with_counter()
        st.session_state.finding_companies_loaded = True


    st.markdown("The natural place to begin is to find companies located in regions where you live or are willing to relocate. As an initial view, Figure 1 shows the cities in our data set where the companies are located, aggregated by state (for a detailed description of the data set, see the `The Data` section). Since our data set focused primarily on Washington, D.C. searches, this area has the most companies. However, D.C. aside, we can see California is the clear runner-up. Zooming into the plot shows that Illinois, New York, and Texas are the next contenders. Notably, states such as Utah, Denver, and Florida have few data science jobs.")



    # begin folium plot
    ######################################################################
    
    # remove missing rows
    location_counts = jobs_results.dropna(subset=['latitude', 'longitude'])

    # init
    folium_plot = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    
    

    # add marker to each location
    folium_marker_cluster = MarkerCluster().add_to(folium_plot)
    for idx, row in location_counts.iterrows():
        folium.Marker(
            [row['latitude'], row['longitude']], 
            tooltip=row['location'],
            icon=folium.Icon(color='#67001E')
        ).add_to(folium_marker_cluster)

    folium_col1, folium_col2, folium_col3 = st.columns([1, 6, 1])

    #with folium_col1:
    #    st.write("")

    #with folium_col2:
    st.markdown('#### Map of Job Locations Density')
    folium_static(folium_plot)
    st.caption('Figure 1: Folium plot showing the density of database jobs in each city, initially aggregated by state.')


    #with folium_col3:
    #    st.write("")

    # callout
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

    # title
    st.markdown('#### Data Science Jobs Table')

    # state selection
    state_selected = st.selectbox("Select a state:", unique_states)

    # get the dataframe for the selected state
    df = tables[state_selected]

    # display the dataframe as a table without row numbers (index)
    st.dataframe(df.reset_index(drop=True), use_container_width=True)

    st.caption('Table 1: Table displaying the jobs in the database, with a dropdown to select by state and columns that can be sorted ascending and descending by clicking.')


    # callout
    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>Tip: Click on a column header to sort the table.</p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("Lastly, to further motivate finding the right employer, in Figure 2, we can see the most common benefits in our Data Science job data set. Data Science is a well-compensating field at the present time. Still, it’s also very competitive, and hence finding the right company to work for will impact how many of the benefits shown below a potential candidate will have access to through their compensation package.")




    # begin word cloud
    ######################################################################

    # comb all descriptions into a single list of words
    all_words = []
    for description in jobs_results['benefits']:
        words = preprocess_text(description, custom_list_of_phrases, custom_list_of_stopwords, custom_list_of_equivalent_phrases)
        all_words.extend(words)

    # word freq count
    word_freq = Counter(all_words)

    # generate wordcloud
    top_words = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50])
    
    word_cloud_ob = WordCloud(background_color='white', max_words=50, width=800, height=400, colormap='RdGy').generate_from_frequencies(top_words)


    # show
    folim_col1, folim_col2, folim_col3 = st.columns([1,6,1])

    with folim_col1:
        st.write("")

    with folim_col2:
        st.markdown('#### Job Benefits Text Word Cloud')
        st.image(word_cloud_ob.to_array(), use_column_width=True)
        st.caption('Figure 2: Word Cloud showing the highest frequency words in the benefits column of the database after text preprocessing.')


    with folim_col3:
        st.write("")


    st.markdown("After finding good companies, we now turn our attention to getting resumes polished in the next section.")


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
    
    if "resume_keywords_loaded" not in st.session_state:
        scroll_to_top_with_counter()
        st.session_state.resume_keywords_loaded = True
    


    st.markdown("First, it is crucial to understand that the field is very competitive and that getting the details right can make a significant difference. Figure 3 shows a network with nodes depicting 20 randomly selected job postings from the data set. Each edge represents the cosine similarity between the words in the job `responsibilities` column after `CountVectorizer` processing. Additionally, the nodes are colored based on `AgglomerativeClustering` from `sklearn`. Interestingly, jobs in the same cluster are not necessarily strongly linked to each other by cosine similarity. This observation highlights that there are important nuances within each job, regardless of being in the same industry or even the same cluster. More than that, it argues against a single resume approach to job applications. It favors tailoring the resume to the particular job posting of interest so it contains the right keywords and maximizes the chances of passing the resume ranking software filters.")




    # begin word network
    ######################################################################


    # remove empty rows
    jobs_results_responsibilities = jobs_results.dropna(subset=['responsibilities'])
    jobs_results_responsibilities.reset_index(drop=True, inplace=True)


    # randomly sample 20
    sample_size = 20
    jobs_results_responsibilities = jobs_results_responsibilities.sample(n=sample_size, random_state=42)
    jobs_results_responsibilities.reset_index(drop=True, inplace=True)


    # stopwrods
    stop_words = set(stopwords.words('english'))

    # run count vectorizer
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(jobs_results_responsibilities['responsibilities'])

    X = X.toarray() # sparse matrix to a dense matrix

    # get cosine similarities
    cosine_similarities = cosine_similarity(X)

    # use sklearn agglomerative clustering for colors
    n_clusters = 4
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='average')
    cluster_labels = cluster_model.fit_predict(1 - cosine_similarities)

    # init network x
    G = nx.Graph()

    # add nodes
    for i in range(len(jobs_results_responsibilities)):
        G.add_node(i, label=jobs_results_responsibilities.loc[i, 'title'], company=jobs_results_responsibilities.loc[i, 'company_name'])

    # add edges
    #for i in range(len(jobs_results_responsibilities)):
    #    for j in range(i+1, len(jobs_results_responsibilities)):
    #        G.add_edge(i, j, weight=cosine_similarities[i][j])
    
    # add edge info
    edges = []
    for i in range(len(jobs_results_responsibilities)):
        for j in range(i+1, len(jobs_results_responsibilities)):
            edge = {
                'source': i,
                'target': j,
                'weight': cosine_similarities[i][j]
            }
            edges.append(edge)


    # Calculate the positions of the nodes using the spring layout
    pos = nx.spring_layout(G, k=5, seed=42)

    # get x and y coords
    x_coords = [pos[i][0] for i in range(len(pos))]
    y_coords = [pos[i][1] for i in range(len(pos))]

    # create plotly object
    fig = go.Figure()

    # add edges lines
    for edge in edges:
        x0, y0 = pos[edge['source']]
        x1, y1 = pos[edge['target']]
        edge_weight = edge['weight']
        fig.add_trace(go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode='lines',
            line=dict(width=edge_weight * 5, color='gray'),  # Use 'edge_weight' as the thickness of the edges
            showlegend=False,
            hoverinfo='none'
        ))

    # add scatterplot trace
    fig.add_trace(go.Scatter(x=x_coords, y=y_coords, mode='markers', marker=dict(size=20, color=cluster_labels, colorscale='RdGy', showscale=False), text=[f"{data['label']}<br>{data['company']}" for i, data in G.nodes(data=True)], hoverinfo='text', showlegend=False))

    # update layour
    fig.update_layout(
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    hovermode='closest',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=30, b=0))


    # show
    network_viz_col1, network_viz_col2, network_viz_col3 = st.columns([1,6,1])

    #with network_viz_col1:
    #    st.write("")

    #with network_viz_col2:
    st.markdown('#### Similarities in Job Responsibilities Detail')
    st.plotly_chart(fig)
    st.caption('Figure 3: Graph of 20 randomly selected jobs related by the similarity of their responsibility description, colored by Agglomerative Clustering.')


    #with network_viz_col3:
    #    st.write("")


    # callout
    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>Tip: Hover over the nodes to see the job title and company</p>
        </div>
        """, unsafe_allow_html=True)




    st.markdown("Next, in Figure 4 below, we can see a tag cloud showing the most frequent words in the `qualifications` columns of our data set, with a dropdown to filter for each job category. Depending on the job that interests the reader, these are essential keywords to include in a base resume. Incorporating them as a default will save a significant amout of time when tailoring a resume for a specific job in the discipline, as many of the keywords will already have been strategically placed.")




    # begin tag cloud
    ######################################################################

    st.markdown('#### Job Qualifications Text Tag Cloud')
    
    # get the category
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = jobs_results["category"].iloc[0]

    categories = sorted(jobs_results["category"].unique())
    
    category = st.selectbox("Select a category:", categories, index=categories.index(st.session_state.selected_category), key='selected_category_widget')
    if category != st.session_state.selected_category:
        st.session_state.selected_category = category
        st.experimental_rerun()

    st.session_state.selected_category = category

    # filter on selected category
    filtered_data = jobs_results[jobs_results["category"] == st.session_state.selected_category]

    # process words with the preprocess function
    all_words = []
    for desc in filtered_data['qualifications']:
        words = preprocess_text(desc, compound_phrases, custom_stopwords, custom_list_of_equivalent_phrases)
        all_words.extend(words)

    # count frequencies
    word_freq = Counter(all_words).most_common(50)


    # show
    tag_cloud_html = generate_tag_cloud_html(word_freq)
    st.markdown(tag_cloud_html, unsafe_allow_html=True)
    st.caption('Figure 4: Tag Cloud showing the 50 most used words, after text preprocessing, in the qualifications field of the database for a selected category.')


    st.markdown("Figure 4 shows several popular requirements for all categories, such as Python, C, and algorithms. Conversely, we can also see specialist skills such as TensorFlow and PyTorch in the Machine Learning category and Smart Contracts and Ethereum in the Blockchain category.")


    st.markdown("Next, to assist the reader, we offer the tool below in which a resume can be provided, and it will check if the keywords as per the selected category are present. To use the tool, a user can simply paste the resume text into the textbox and click the `Check Missing Keywords` button.")




    # begin missing words check
    ######################################################################

    st.markdown('#### Resume Keywords Check Tool')

    # input resume 
    input_text = st.text_area("Enter resume text:")

    # use words from tag cloud
    word_list = [word for word, _ in word_freq]

    # run the check
    if st.button("Check Missing Keywords"):
        
        # check if not empty
        if input_text and word_list:
            missing_words = check_missing_words(input_text, word_list)
            
            # display missing words
            if missing_words:
                st.markdown(f"<p style='color: grey;'>The following words are missing from the text: {', '.join(missing_words)}</p>", unsafe_allow_html=True)
            # if all present
            else:
                st.markdown("<p style='color: grey;'>All words are present in the text.</p>", unsafe_allow_html=True)
        # if no text in input
        else:
            st.markdown("<p style='color: grey;'>Please enter resume text.</p>", unsafe_allow_html=True)
            
            
    st.markdown(
        """
        <div class="bd-callout bd-callout-info">
             <p>Don't have a resume and want to try the tool? Try the following text: <mark>I am proficient in data, analysis, and statistics</mark>. Add or remove words as needed to compare with the Tag Cloud.</p>
        </div>
        """, unsafe_allow_html=True)



    st.markdown("Now that we have covered these critical aspects of a job search, we can move on to some concluding remarks.")


    # page transitions
    ######################################################################
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
    
    scroll_to_top_with_counter()

    st.markdown("In this project, we began by motivating the importance of being selective about which companies to apply to and getting a resume that stands out, particularly for resume ranking software. Then, we discussed the literature, covering the importance of having clear goals before beginning the application process, maintaining a manageable workload, and keeping track of successes. We also discussed the six steps for the job search process and motivated focusing on company search resume polishing (steps two and three). Next, we presented several visualizations and tools using our data set to evaluate companies, including their location and common benefits offered to Data Science applicants. Lastly, we covered why having a resume that stands out, and we visualized the most important keywords for each Data Science job category is essential. Finally, we concluded by offering a tool for comparing the common keywords against a resume.")
    
    st.markdown("Future versions of this project could include targeting additional aspects of the hiring process of the job search process discussed in the literature. Similarly, the data set could be expanded to contain significantly more jobs, allowing for richer statistics. Also, alternative methods other than cosine similarity could be attempted when comparing jobs. Lastly, the keyword check tool could be expanded to allow for inputting a job description and estimating how likely it is to get a high score in resume ranking software.")
    
    st.markdown("Thank you for reading and best of luck with your job search!")
    
    
    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        image = Image.open('Code/Streamlit/Images/pexels-pixabay-327540.jpg')
        st.image(image, caption='Photo credit: Pexels (2017)', width=450)

    with col3:
        st.write("")



    # page transitions
    ######################################################################       
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
    
    scroll_to_top_with_counter()
    
    st.markdown("The data was obtained from Georgetown MS in Data Science and Analytics (2023). The data tables produced during the cleaning and preparation process are available in the [Data](https://github.com/vdelimad/returning-student-scholarship/tree/main/Data) folder in GitHub.")
    st.markdown("#### Data Cleaning")
    st.markdown("The raw data consisted of 85 json files with data gathered from searches for Data Science jobs of different categories for the D.C. area and the United States in general. The cleaning process consisted of parsing the json files into `Pandas` data frames, which are more easily processable with `Python` code. The cleaning exercise and EDA are available in the [DataCleaning](https://github.com/vdelimad/returning-student-scholarship/tree/main/Code/DataCleaning) folder on GitHub. A total of 591 job postings made up the data set after completing the cleaning and removing duplicates.")
    st.markdown("#### Data Preparation")
    st.markdown("Data preparation included using the `Nominatim` library from `Geopy` to obtain the coordinates of the cities to construct the `Folium` plot. Additional preparation included preprocessing the text data with a custom list of phrases, stop words, and equivalent phrases, as well as other typical methods for cleaning text data. The processed text was used to make the Tag Cloud and Word Cloud. Lastly, `CountVectorizer` was used to generate vectors from text that could be analyzed with cosine similarity to build the network plot.")




def references_page():

    st.markdown("## References")
    
    scroll_to_top_with_counter()
    
    st.markdown("Amabile T. 2014. The progress principle: using small wins to ignite joy, engagement, and creativity at work. Brilliance Audio.")
    st.markdown("Ceniza-Levine C, Thanasoulis-Cerrachio C. 2011. Six steps to job search success. Arlington, Virginia: Saylor Foundation.")
    st.markdown("Dalton S. 2012. The 2-hour job search: using technology to get the right job faster. 1st ed. Berkeley: Ten Speed Press.")
    st.markdown("Georgetown MS in Data Science and Analytics. 2023. DSAN 2023 Scholarship: 2023-04-14-Data-job-search. [accessed 2023 May 2]. https://drive.google.com/drive/folders/1UYeTLu9fV2mRYOvKrNvnrH18Csu0AMkg?usp=sharing.")
    st.markdown("Pexels. 2017. Close-up of Human Hand. [accessed 2023 May 8]. https://www.pexels.com/photo/close-up-of-human-hand-327540/.")
    st.markdown("Resume Worded. 2023 May 8. Targeted Resume. Resume Worded. [accessed 2023 May 8]. https://resumeworded.com.")
    st.markdown("Rothman J. 2014. Manage your job search. Arlington, MA Practical Ink.")
    st.markdown("Small Business & Entrepreneurship Council. 2023. Facts & Data on Small Business and Entrepreneurship. [accessed 2023 May 8]. https://sbecouncil.org/about-us/facts-and-data/.")
    st.markdown("Winkle M. 2020. A Clipboard Near Pen and Laptop On a Marble Surface. https://www.pexels.com/photo/a-clipboard-near-pen-and-laptop-on-a-marble-surface-4101343/.")

  


################################################
# streamlit setup
################################################


# init streamlit
st.set_page_config(page_title="The Data Science Job Search Survival Guide", page_icon = 'Code/Streamlit/Images/handshake.png')
button_clicked = False

# init default page
default_page = "Introduction"
page = default_page

# add custom css
with open('Code/Streamlit/custom.css') as f:
    st.markdown (f'<style>{f.read()}</style>', unsafe_allow_html=True)





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
    resume_keywords_page(jobs_results, custom_list_of_phrases, custom_list_of_stopwords)

elif st.session_state.selected_page == "Conclusions":
    conclusions_page()

# about section
elif st.session_state.selected_page == "The Data":
    the_data_page()

elif st.session_state.selected_page == "References":
    references_page()

