'''
Partially sourced from ChatGPT (GPT-4) to help acquire our dataset. We ended up
not using this since we found a good dataset on Kaggle, but this works.
'''

from datetime import datetime, timezone
from googleapiclient.discovery import build
from constants import API_KEY

# Set up the API credentials
# creds, project_id = google.auth.default()
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Specify the video IDs to look up
video_ids = ['lOg1yv8suZM', '5TCGXyHzSSc', 'oave4Z939as']

# Use this to search for videos in a particular genre
# search_response = youtube.search().list(
#     part='id',
#     q='gaming',
#     type='video',
#     maxResults=500
# ).execute()

# Extract video IDs from the search response
# video_ids = [item['id']['videoId'] for item in search_response['items']]

# Not sure why but this does not like going over 50 at a time
max_results = 50

# Retrieve a list of videos sorted by date published
search_response = youtube.search().list(
    type='video',
    order='date',
    part='id',
    maxResults=max_results
).execute()

videos_response = youtube.videos().list(
    part='snippet,statistics',
    id=','.join(video_ids)
).execute()
count = 1

# Iterate over the video responses to extract the desired information
for video in videos_response['items']:
    # Extract video information
    video_id = video['id']
    title = video['snippet']['title']
    thumbnail_url = video['snippet']['thumbnails']['default']['url']
    view_count = video['statistics']['viewCount']
    like_count = video['statistics']['likeCount']
    date_published = datetime.strptime(video['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    
    channel_id = video['snippet']['channelId']
    channel_response = youtube.channels().list(
    id=channel_id,
    part='statistics'
    ).execute()
    subscriber_count = channel_response['items'][0]['statistics']['subscriberCount']

    # Print the extracted information
    print(f'Video ID: {video_id}')
    print(f'Title: {title}')
    print(f'Thumbnail URL: {thumbnail_url}')
    print(f'View count: {view_count}')
    print(f'Like count: {like_count}')
    print(f'Date published: {date_published}')
    print(f'Subscribers: {subscriber_count}')
    print('--------------------------  ' + str(count))
    count +=1 