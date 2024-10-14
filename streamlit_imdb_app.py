import streamlit as st
import pandas as pd
from pinecone import Pinecone
import requests
import os
from pinecone import ServerlessSpec
from pinecone import Pinecone as PineconeClient
import time
from datetime import datetime

try:
  import yaml
  with open('config.yaml') as f:
      data = yaml.load(f, Loader=yaml.FullLoader)
      api_hugging = data['API_HUGGING']
      pinecone_api_key = data['API_PINECONE']
except Exception as e:
  print(e)
  print('running in stremlit')
  api_hugging = st.secrets["API_HUGGING"]
  pinecone_api_key = st.secrets["API_PINECONE"]

# Set the page layout to wide mode

st.set_page_config(page_title="Movie Search Engine", layout="wide")

index_name = 'movies-embeddings'

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)
pc = Pinecone(api_key=pinecone_api_key, environment=spec)

index = pc.Index(index_name)

# Genre list
genre_list = ['Action', 'Drama', 'Adventure', 'Sci-Fi', 'Animation', 'Crime',
       'Comedy', 'Thriller', 'Fantasy', 'Horror', 'History', 'Mystery',
       'Biography', 'War', 'Western', 'Sport', 'Family', 'Romance',
       'Music', 'Musical', 'Film-Noir', 'Game-Show', 'Adult',
       'Reality-TV']

def query(payload):
	API_URL = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large"
	headers = {"Authorization": "Bearer " + api_hugging,
						"x-wait-for-model": "true"}

	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

# Hugging Face query function
def create_embedding(payload):
    API_URL = "https://api-inference.huggingface.co/models/intfloat/multilingual-e5-large"
    headers = {
        "Authorization": "Bearer " + api_hugging,
        "x-wait-for-model": "true"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def search(query_str, genre, rating, year, voting_count, top_k):
    query_vector = query({
        "inputs": query_str,
    })

    if rating:
        filter_rating = rating
    else:
        filter_rating = 0

    if year:
        filter_year = year
    else:
        filter_year = 0

    if voting_count:
        filter_voting_count = voting_count
    else:
        filter_voting_count = 0

    if genre:
         conditions ={
                "Generes": { "$in": [genre] },
                "Rating": { "$gte": filter_rating },
                "year": { "$gte": filter_year },
                "User Rating": { "$gte": filter_voting_count }
                }
    else:
        conditions ={
                "Rating": { "$gte": filter_rating },
                "year": { "$gte": filter_year },
                "User Rating": { "$gte": filter_voting_count }
                }

    responses = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=conditions
    )

    print(responses)

    # Format the responses for better display
    response_data = []
    for response in responses['matches']:
        response_data.append({
            'Title': response['metadata']['movie title'],
            'Overview': response['metadata']['Overview'],
            'Director': response['metadata']['Director'],
            'Genre': response['metadata']['Generes'],
            'year': int(response['metadata']['year']),
            'Rating': response['metadata']['Rating'],
            'Link': response['metadata']['path'],
        })

    df = pd.DataFrame(response_data)
    return df
# Language toggle
language = st.radio("Language / Idioma", ("English", "Español"))

# Define labels based on the selected language
if language == "English":
    st.title("Movie Search Engine")
    st.write("Enter your query to get the best matches in film history!\n\n")
    
    query_label = "Search terms here! Example: Time traveling adventure."
    rating_label = "Minimum Rating IMDB"
    vote_count_label = "Minimum vote count"
    genre_label = "Movie Genre (optional)"
    year_label = "Minimum Movie Year"
    results_label = "Number of Results"
    search_button_label = "Search"
    results_section_label = "Results"
    
else:
    st.title("Buscador de películas")
    st.write("Introduce tu consulta para conseguir las mejores coincidencias en la historia del cine!")
    
    query_label = "Consulta aquí. Ejemplo: Aventura de viaje en el tiempo"
    rating_label = "Puntuación mínima IMDB"
    vote_count_label = "Minima cantidad de votos"
    genre_label = "Género de la película (opcional)"
    year_label = "Minimo Año de la película"
    results_label = "Cantidad de resultados"
    search_button_label = "Buscar"
    results_section_label = "Resultados"

# Use columns to align the query on the left and sliders on the right, with more separation
col1, col2 = st.columns([1, 1], gap="large")  # Increased the separation by giving col1 more width

with col1:
    # Increased the height of the text area
    query_text = st.text_area(query_label, "", max_chars=500, height=210)  # Larger height for query input
    genre = st.selectbox(genre_label, [""] + genre_list)  # Blank by default

with col2:
    min_rating = st.slider(rating_label, 1, 10, 6)  # Slider for minimum rating
    year = st.slider(year_label, 1930, datetime.now().year, 1980)  # Slider for year
    minimum_votes = st.slider(vote_count_label, 0, 10000, 500)  # Slider for year
    num_results = st.slider(results_label, 1, 50, 10)  # Slider for number of results (limit to 30)

# Function to display results in a card format with two columns
def display_results_in_two_columns(df):
    col1, col2 = st.columns(2)  # Create two columns for displaying results side by side
    
    for i, row in df.iterrows():
        if i % 2 == 0:
            # Display in the first column
            with col1:
                with st.expander(f"{row['Title']} ({row['year']}) - {row['Rating']} ⭐"):
                    st.write(f"**Overview**: {row['Overview']}")
                    st.write(f"**Director**: {row['Director']}")
                    st.write(f"**Genre**: {', '.join(row['Genre'])}")
                    st.write(f"**Year**: {row['year']}")
                    st.write(f"**Rating**: {row['Rating']}")
                    st.markdown(f"[More info](https://www.imdb.com/{row['Link']})", unsafe_allow_html=True)  # Add clickable link
        else:
            # Display in the second column
            with col2:
                with st.expander(f"{row['Title']} ({row['year']}) - {row['Rating']} ⭐"):
                    st.write(f"**Overview**: {row['Overview']}")
                    st.write(f"**Director**: {row['Director']}")
                    st.write(f"**Genre**: {', '.join(row['Genre'])}")
                    st.write(f"**Year**: {row['year']}")
                    st.write(f"**Rating**: {row['Rating']}")
                    st.markdown(f"[More info](https://www.imdb.com/{row['Link']})", unsafe_allow_html=True)  # Add clickable link

if st.button(search_button_label):
  if query_text != '':
    results = search(query_text, genre, min_rating, year, minimum_votes, num_results)
    st.write(results_section_label)
    
    if results.empty:
        st.write("No results found.")
    else:
        # Display results as cards
        display_results_in_two_columns(results)
    
    query_text = ''