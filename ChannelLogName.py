# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

date = datetime.now().strftime("%Y_%m_%d-%I_%p")

# Replace channel_here with the channel you want to track and save the file
# before re-running the bot
channel = 'channel_here' 
log_name = str(f'{channel}_chat_log_{date}')

