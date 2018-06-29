import random
import numpy as np
import pandas as pd
import trading_env

import datetime
import time

# df = pd.read_hdf('dataset/SGXTW.h5', 'STW')
# print(df.head())
# print(df.tail())

# env = trading_env.make(env_id='training_v1', obs_data_len=256, step_len=128,
#                        df=df, fee=0.1, max_position=5, deal_col_name='Price', 
#                        feature_names=['Price', 'Volume', 
#                                       'Ask_price','Bid_price', 
#                                       'Ask_deal_vol','Bid_deal_vol',
#                                       'Bid/Ask_deal', 'Updown'])


def get_market_data(market):
    df = pd.read_html("https://coinmarketcap.com/currencies/" + market + "/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"), flavor='html5lib')[0]
    df = df.assign(Date=pd.to_datetime(df['Date']))
    df = df.rename(columns = {'Date':'datetime'})
    df['Volume'] = (pd.to_numeric(df['Volume'], errors='coerce').fillna(0))
    df.to_hdf('dataset/BITCOIN.h5', 'STW', mode='w')
    return df 

try:
    df = pd.read_hdf('dataset/BITCOIN.h5', 'STW')
except OSError as e:
    df = get_market_data('bitcoin')

df = df.iloc[::-1]
df = df.reset_index(drop=True)
df['serial_number'] = 0

print(df.head())
print(df.tail())

env = trading_env.make(env_id='training_v1', obs_data_len=1, step_len=7,
                       df=df, fee=0.1, max_position=1, deal_col_name='Close**', 
                       feature_names=['Close**', 'Volume', 'Market Cap'])


env.reset()
env.render()

state, reward, done, info = env.step(random.randrange(3))

### randow choice action and show the transaction detail
for i in range(500):
    print(i)
    state, reward, done, info = env.step(random.randrange(3))
    print(state, reward)
    env.render()
    if done:
        break
env.transaction_details
