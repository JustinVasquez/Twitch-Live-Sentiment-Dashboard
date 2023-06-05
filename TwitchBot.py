# -*- coding: utf-8 -*-

from twitchio.ext import commands
from colorama import Fore
#import asyncio
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from ChannelLogName import log_name
from MessageCleaning import str_cleaner


timestamp = []
chatter = []
chat_log = []
p_n = []


class TwitchBot(commands.Bot):

    def __init__(self, token, channel, classifier):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot...
        super().__init__(token=token, prefix='?', initial_channels=[channel])
        self.classifier = classifier
        self.channel = channel
        self.fig, self.axs = plt.subplots(3)
        self.df = pd.DataFrame(data = [], index=None, columns=['timestamp','chatter','chat','sentiment','avg_sent'])
        print('Initialized')

    async def event_ready(self):
        # We are logged in and ready to chat and use commands...
        print(f'Logged in as | {self.nick}')


    async def event_message(self, message):
        if message.echo:
            return 
        timestamp.append(message.timestamp)
        chatter.append(message.author.name)
        chat_log.append(message.content)
        sentiment = self.classifier.predict([str_cleaner(message.content)])
        p_n.append(sentiment)
        
        new_timestamp = [message.timestamp]
        new_df = pd.DataFrame({'timestamp':new_timestamp, 'chatter':message.author.name, 'chat':message.content, 'sentiment':sentiment, 'avg_sent':0})
        self.df = pd.concat([self.df, new_df], ignore_index = True)
        
        if (sentiment == -1):
            print(Fore.LIGHTRED_EX + f'{message.timestamp} Author: {message.author.name}, Sent: {message.content}, Sentiment: {sentiment}')
            print(str_cleaner(message.content))
        elif (sentiment == 0):
            print(Fore.WHITE + f'{message.timestamp} Author: {message.author.name}, Sent: {message.content}, Sentiment: {sentiment}')
            print(str_cleaner(message.content))
        else:
            print(Fore.GREEN + f'{message.timestamp} Author: {message.author.name}, Sent: {message.content}, Sentiment: {sentiment}')
            print(str_cleaner(message.content))
        print('Messages gathered: ', len(chat_log), '\n')
        
        await self.handle_commands(message)
        
        # We can uncomment the plt.style line to use a different MPL visualization style like ggplot 
        
        if len(chat_log) >= 0:
            #plt.style.use('ggplot')
            self.df.iloc[-1, -1] = self.df.iloc[-20:]['sentiment'].mean()
            self.df.to_csv(log_name + ".csv")
            
            self.axs[0].clear()
            self.axs[1].clear()
            self.axs[2].clear()
            
            df = self.df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = df['timestamp'].apply(lambda x: x - timedelta(hours=5))
            df.dropna(axis=0, inplace=True)
            
            active_chatters = df.groupby('chatter').chat.count().sort_values(ascending=False)
            #colors = sns.color_palette('pastel')
            bar_x = active_chatters.index[:10] 
            bar_y = active_chatters[:10]

            time_idx = df.set_index('timestamp')
            msg_per_min = time_idx['chat'].resample('1T').count()

            self.axs[0].bar(x=bar_x, height=bar_y, color = 'purple')
            self.axs[0].set_title('Messages per Chatter')
            self.axs[0].set_xticks(ticks = bar_x)
            self.axs[0].set_xticklabels(labels = bar_x, rotation=-45, ha='center')
            self.axs[0].grid(which='major', axis='y', color='gray')
            self.axs[0].set_axisbelow(True)

            self.axs[1].plot(df['timestamp'], df['avg_sent'], color = 'purple')
            self.axs[1].set_title('Chat Average Sentiment')
            self.axs[1].grid(which='major', axis='y', color='gray')
            self.axs[1].set_axisbelow(True)
            
            self.axs[2].plot(msg_per_min.index, msg_per_min, color = 'purple')
            self.axs[2].set_title('Messages per Minute')
            self.axs[2].grid(which='major', axis='y', color='gray')
            self.axs[2].set_axisbelow(True)
            
            plt.tight_layout(pad=3.0)

            plt.pause(0.1)
            
            plt.show(block=False)
            
            plt.savefig(log_name)
            
            pass
        

              

