# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import os
from TwitchBot import TwitchBot
from transformers import pipeline
from ChannelLogName import channel
import joblib


# Allocate a pipeline for sentiment-analysis

# The first classifier variable is set to a pre-trained classifier, imported 
# from the HuggingFace transofmers library. It was used initially to test
# and compare if the classifier would be suffice for the project, but was not 
# effective enough as it was trained on Twitter data rather than Twitch messages.

#classifier = pipeline('sentiment-analysis')

classifier = joblib.load('fit_lr.pkl')

# Accesses the .env file that holds your Twitch access token
load_dotenv()
token = os.getenv('TWITCH_ACCESS_TOKEN')

# Specifies which channel to connect to
channel = channel

bot = TwitchBot(token, channel, classifier)
bot.run()
