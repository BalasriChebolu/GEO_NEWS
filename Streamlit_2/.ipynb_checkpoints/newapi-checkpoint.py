import pandas as pd
import requests
import datetime
import spacy
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from pymongo import MongoClient
from pymongo.server_api import ServerApi

NEWSAPI_KEY = '822d31ec7b0b48ac8a057ecdf8974fed'
NEWSAPI_ENDPOINT = 'https://newsapi.org/v2/everything'

disaster_keywords = ['earthquake', 'flood', 'tsunami', 'hurricane', 'wildfire', 'forestfire', 'tornado', 'cyclone', 'volcano', 'drought', 'landslide', 'storm', 'blizzard', 'avalanche', 'heatwave']

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Initialize geocoder
geolocator = Nominatim(user_agent="my_geocoder")

def fetch_live_data(keyword):
    # Calculate the date 2 days ago
    two_days_ago = datetime.datetime.now() - datetime.timedelta(days=4)
    
    params = {
        'apiKey': NEWSAPI_KEY,
        'q': keyword,
        'from': two_days_ago.strftime('%Y-%m-%d'),  # From 1 days ago
        'to': datetime.datetime.now().strftime('%Y-%m-%d'),  # To today
        'language': 'en',
    }

    response = requests.get(NEWSAPI_ENDPOINT, params=params)
    return response.json().get('articles', [])

def identify_disaster_event(title):
    if title is None:
        return 'Unknown'
    
    # Identify the type of disaster event based on keywords
    for keyword in disaster_keywords:
        if keyword.lower() in title.lower():
            return keyword.capitalize()

    return 'Unknown'

def extract_location_ner(text):
    doc = nlp(text)
    location_ner_tags = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
    return location_ner_tags

def get_coordinates(location):
    try:
        location_info = geolocator.geocode(location, timeout=10) # Increase timeout if needed
        if location_info:
            return location_info.latitude, location_info.longitude
        else:
            return (np.nan, np.nan)
    except GeocoderTimedOut:
        print(f"Geocoding timed out for {location}")
        return (np.nan, np.nan)
    except Exception as e:
        print(f"Error geocoding {location}: {str(e)}")
        return (np.nan, np.nan)

if __name__ == "__main__":
    all_live_data = []
    for keyword in disaster_keywords:
        live_data = fetch_live_data(keyword)
        for article in live_data:
            published_at = article.get('publishedAt', datetime.datetime.utcnow())
            disaster_event = identify_disaster_event(article['title'])
            filtered_article = {
                'title': article['title'],
                'disaster_event': disaster_event,
                'timestamp': published_at,
                'source': article['source'],
                'url': article['url']
            }
            
            all_live_data.append(filtered_article)
    
    df = pd.DataFrame(all_live_data)

    df['disaster_event'].replace(to_replace="Unknown", value=np.nan, inplace=True)
    df.dropna(axis=0, inplace=True)
    df.drop_duplicates(subset='title', inplace=True)
    df['source'] = df['source'].apply(lambda x: x['name'])

    
    df['location_ner'] = df['title'].apply(extract_location_ner)
    df['location_ner'] = df['location_ner'].apply(lambda x: np.nan if len(x) == 0 else x)
    df.dropna(axis=0, inplace=True)

    def fun(text):
        country, region, city = '', '', ''
        if len(text) == 1:
            country = text[0]
        elif len(text) == 2:
            country, region = text[0], text[1]
        elif len(text) == 3:
            country, region, city = text[0], text[1], text[2]
        return country, region, city

    a = df['location_ner'].apply(fun)

    df['Country'] = ''
    df['Region'] = ''
    df['City'] = ''

    df[['Country', 'Region', 'City']] = pd.DataFrame(a.tolist(), index=df.index)

    def create_location(row):
        if row['City']:
            return row['City']
        elif row['Region']:
            return row['Region']
        else:
            return row['Country']

    df['Location'] = df.apply(create_location, axis=1)
    df = df.dropna(subset=['Location'])

    exclude_locations = ['avalanche', 'blizzard', 'cyclone', 'drought', 'earthquake', 
                         'flood', 'heatwave', 'hurricane', 'landslide', 'storm', 
                         'tornado', 'tsunami', 'volcano', 'wildfire', 'hockey', 'a.i.']

    df = df[~df['Location'].str.lower().isin(exclude_locations)]
    df = df[~df['url'].str.lower().str.contains('politics|yahoo|sports|entertainment|cricket|fark.com|mondoweiss|latimes')]
    df = df[~df['source'].str.contains('Slashdot.org|Biztoc.com')]

    df['Coordinates'] = df['Location'].apply(get_coordinates)

    df[['Latitude', 'Longitude']] = pd.DataFrame(df['Coordinates'].apply(lambda x: x if x else (np.nan, np.nan)).tolist(), index=df.index)

    df.drop('Coordinates', axis=1, inplace=True)

    df = df.dropna(subset=['Latitude', 'Longitude'])

    # MongoDB Atlas connection URI
    uri = "mongodb+srv://aryanrvimpadapu:MUTBZgApDRVxxIXY@cluster0.fs4he7a.mongodb.net/?retryWrites=true&w=majority"

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # Access the GeoNews database and disaster_info collection
    db = client["GeoNews"]
    collection = db["disaster_info"]

    # Create a DataFrame

    # Convert DataFrame to a list of dictionaries
    data_list = df.to_dict(orient='records')

    # Insert the data list into the collection
    try:
        result = collection.insert_many(data_list)
        print("Documents inserted successfully. IDs:", result.inserted_ids)
    except Exception as e:
        print("An error occurred:", e)

    pipeline = [
        {"$group": {
            "_id": "$title",
            "uniqueIds": {"$addToSet": "$_id"},
            "count": {"$sum": 1}
        }},
        {"$match": {
            "count": {"$gt": 1}
        }}
    ]

    duplicates = list(collection.aggregate(pipeline))

    # Step 2: Remove duplicates
    for duplicate in duplicates:
        uniqueIds = duplicate["uniqueIds"]
        # Keep the first document and remove the rest
        for _id in uniqueIds[1:]:
            collection.delete_one({"_id": _id})

    print("Duplicates removed successfully.")