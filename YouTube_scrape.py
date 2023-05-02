from datetime import datetime, timezone
from googleapiclient.discovery import build
from constants import API_KEY

# Set up the API credentials
# creds, project_id = google.auth.default()
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Specify the video ID
# video_id = 'lOg1yv8suZM'
video_id = 'lOg1yv8suZM'

# Use the videos().list() method to retrieve information about the video
# Video IDs to look up
video_ids = ['lOg1yv8suZM', '5TCGXyHzSSc', 'oave4Z939as']

# # Call the videos().list method to retrieve information about each video

#USE FOR SEARCHING FOR VIDS
# search_response = youtube.search().list(
#     part='id',
#     q='gaming',
#     type='video',
#     maxResults=500
# ).execute()

max_results = 50 #not sure why but this hates going over 50

# Retrieve a list of videos sorted by date published
search_response = youtube.search().list(
    type='video',
    order='date',
    part='id',
    maxResults=max_results
).execute()


# THIS TOO FOR SEARCHING
# Extract video IDs from the search response
video_ids = [item['id']['videoId'] for item in search_response['items']]

videos_response = youtube.videos().list(
    part='snippet,statistics',
    id=','.join(video_ids) #change to video_ids if want more
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