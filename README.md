# Twitch-Chat-Live-Sentiment
Analyzing a Twitch channel's chat sentiment in realtime (Currently in progress)
## Context
Over the course of the pandemic, I began watching Twitch live streams as I found them to be a refreshing change of pace akin to a live podcast rather than a typical TV show. Apart from the rare prerecorded clip usually in collaboration with a sponsor, the content streamers broadcast is live along with genuine reactions to the games they playing, the videos they watch, and the other creators they interact with. As such, I wanted to track how viewers, colloquially known as "chat", react in realtime to the content being presented. With this, I'd like to assist up-and-coming streamers identify and capitalize on actions/content in their stream that leads to higher positive chat interaction by cross-referencing this when reviewing their streams.  

## Dataset
I utilized Twitch's API, TwitchIO, to connect to channels and save messages being sent in chat to train my model, as well as to connect for live visual and sentiment analysis. 

## Analysis Process
1. Collecting chat messsages and manually classifying (positive, neutral, negative)
2. Data cleaning
3. Exploratory Data Analysis which creates the basis for the live graphs 
4. Training sentiment analysis model 
5. Defining functions to produce live graphs for chat sentiment 

## File Descriptions
1. ChatBot.py - Main file that runs the subsequent files, also specifies which channel to connect to 
2. ChatConnection.py - This initiates the connection to the Twitch channel's chat to capture the messages sent by viewers
3. ChatProcessing.py - Contains the functions used in ChatConnection.py to produce live graphs
4. ChatSentimentAnalyzer.py - This contains the model that was trained with previously collected messages and used for live analysis
5. MessageCleaner.py - Contains two functions that clean the imported text
6. .env - Contains your Twitch access token needed in order to connect to TwitchIO
