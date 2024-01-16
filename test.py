import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date
import pyotp
import json
from api_helper import ShoonyaApiPy

#  Login - api credentials
uid = 'FA188643'
pwd = 'Yadnesh@123'
token = '5H7VVU44RJMBW2233RGOWD236VF323MD'
factor2 = pyotp.TOTP(token).now()
vc = 'FA188643_U'
app_key = 'a5e1164b454fb0c4b75ada3160152cb4'
imei = 'abc1234'
# api object
api = ShoonyaApiPy()
ret = api.login(userid=uid, password=pwd, twoFA=factor2, vendor_code=vc, api_secret=app_key, imei=imei)

def main(Token, TradingSymbol):
    # Collecting the Data for every 5min
    data2 = api.get_time_price_series(exchange='NSE', token=Token, starttime=1688183100, interval=5)
    df=pd.DataFrame(data2)

    # Necessary formating of the dataframe
    def update_into_column(data):
        data['into'] = pd.to_numeric(data['into'], errors='coerce')
        data['inth'] = pd.to_numeric(data['inth'], errors='coerce')
        data['intl'] = pd.to_numeric(data['intl'], errors='coerce')
        data['intc'] = pd.to_numeric(data['intc'], errors='coerce')
        data['intv'] = pd.to_numeric(data['intv'], errors='coerce')
        data['time'] = pd.to_datetime(data['time'], format='%d-%m-%Y %H:%M:%S') 
        data = data.iloc[::-1].reset_index(drop=True)
        for index, row in data.iterrows():
            if row['time'].strftime('%H:%M:%S') == '09:15:00':
                if index > 0:
                    data.at[index, 'into'] = data.at[index - 1, 'intc']
        return data

    df = update_into_column(df)


    df = df[['stat', 'time', 'into', 'inth', 'intl', 'intc', 'intv']]


    # Intervals Function to make dataframe for required intervals
    def interval_buckets(df, interval):
        n = int(interval/5)
        new_data = []
        
        for i in range(0, len(df), n):
            temp_df = df.iloc[i:i+n]
            
            
            if len(temp_df) >= 1:
                datetime = temp_df.iloc[0]['time']
                time = temp_df.iloc[0]['time']
                into = temp_df.iloc[0]['into']
                intc = temp_df.iloc[-1]['intc']
                vol = temp_df['intv'].sum()
                
                inth = temp_df['inth'].max()
                intl = temp_df['intl'].min()
                
                new_data.append({'datetime': datetime, 'into': into, 'intc': intc, 'inth': inth, 'intl': intl, 'vol':vol})

        new_df = pd.DataFrame(new_data)

        return new_df


    # Dataframe for 125min interval
    int125_df = interval_buckets(df, 125)
    int125_df = int125_df.tail(200)
    int125_df = int125_df.reset_index(drop=True)
    # int125_df

   
    # Dataframe for daily interval
    daily_df = interval_buckets(df, 375)
 

    # Dataframe for 15min interval
    int15_df = interval_buckets(df, 15)
    int15_df = int15_df.tail(200)
    int15_df = int15_df.reset_index(drop=True)
    

    # Function for checking candle type (Bearish/Bullish)
    def type(row):
        open_price = float(row['into'])
        close_price = float(row['intc'])
        if open_price > close_price:
            return "Bearish"
        else:
            return "Bullish"
        
    # Function for calculating atr (Used for determining candle type)
    def calculate_atr(dataframe):
        # Convert columns to appropriate numerical types
        dataframe['into'] = pd.to_numeric(dataframe['into'])
        dataframe['intc'] = pd.to_numeric(dataframe['intc'])
        dataframe['inth'] = pd.to_numeric(dataframe['inth'])
        dataframe['intl'] = pd.to_numeric(dataframe['intl'])

        # Calculate True Range (TR)
        dataframe['high_low'] = dataframe['inth'] - dataframe['intl']
        dataframe['high_close'] = abs(dataframe['inth'] - dataframe['intc'].shift())
        dataframe['low_close'] = abs(dataframe['intl'] - dataframe['intc'].shift())
        dataframe['true_range'] = dataframe[['high_low', 'high_close', 'low_close']].max(axis=1)

        # Calculate Average True Range (ATR)
        dataframe['atr'] = dataframe['true_range'].rolling(window=14).mean()

        # Fill NaN values in the 'atr' column with the threshold value
        dataframe['atr'] = dataframe['atr'].fillna(dataframe['true_range'])

        # Drop temporary columns used for TR calculation
        dataframe = dataframe.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1)

        return dataframe

    # Function for checking candle type (ERC/NRC)
    def determine_candle_type(row):
        range_value = row['inth'] - row['intl']

        body_size = abs(row['into'] - row['intc'])

        if range_value > 0.7 * row['atr'] and body_size >= 0.5*range_value:
            return 'ERC'
        else:
            return 'NRC'


    # Function for Determining zone-patterns and zone-types
    def zone_pattern(df):

        df['into'] = pd.to_numeric(df['into'], errors='coerce')
        df['inth'] = pd.to_numeric(df['inth'], errors='coerce')
        df['intl'] = pd.to_numeric(df['intl'], errors='coerce')
        df['intc'] = pd.to_numeric(df['intc'], errors='coerce')
        # average_difference = (df['inth'] - df['intl']).mean()
        df = calculate_atr(df)
        
        df['Classification'] = df.apply(determine_candle_type, axis=1)

        
        df["type"] = df.apply(type, axis=1)

        zones = []

        index = 0  
        while index < len(df) - 1:
            if df.at[index, 'Classification'] == 'ERC' and df.at[(index+1),"Classification"] == "NRC":
                e1_startdate = df.at[index, 'datetime']
                e1_type = df.at[index, 'type']
                count_nrc = 0


                for i in range(index + 1, len(df)):
                    if count_nrc <= 7 and df.at[i, 'Classification'] == 'NRC':
                        count_nrc += 1
                    elif count_nrc <= 7 and df.at[i, 'Classification'] == 'ERC':
                        e2_startdate = df.at[i, 'datetime']
                        e2_type = df.at[i, 'type']

                        # zone_high = df.loc[(index+1):(i-1), ['into', 'intc']].max().max()
                        # zone_low = df.loc[(index+1):(i-1), ['into', 'intc']].min().min()

                        if e1_type == 'Bearish' and e2_type == 'Bearish':
                            zone_type = 'supply'
                            zone_pattern = 'dbd'
                            zone_high = df.loc[(index):(i), 'inth'].max()
                            zone_low = df.loc[(index+1):(i-1), ['into', 'intc']].min().min()

                        elif e1_type == 'Bearish' and e2_type == 'Bullish':
                            zone_type = 'demand'
                            zone_pattern = 'dbr'
                            zone_high = df.loc[(index+1):(i-1), ['into', 'intc']].max().max()
                            zone_low = df.loc[(index):(i), 'intl'].min()

                        elif e1_type == 'Bullish' and e2_type == 'Bearish':
                            zone_type = 'supply'
                            zone_pattern = 'rbd'
                            zone_high = df.loc[(index):(i), 'inth'].max()
                            zone_low = df.loc[(index+1):(i-1), ['into', 'intc']].min().min()

                        elif e1_type == 'Bullish' and e2_type == 'Bullish':
                            zone_type = 'demand'
                            zone_pattern = 'rbr'
                            zone_high = df.loc[(index+1):(i-1), ['into', 'intc']].max().max()
                            zone_low = df.loc[(index+1):(i), 'intl'].min()

                        zones.append({
                            'leg-in-datetime': e1_startdate,
                            'leg-out-datetime': e2_startdate,
                            'zonehigh': zone_high,
                            'zonelow': zone_low,
                            'basecount' : count_nrc,
                            'zonepattern' : zone_pattern,
                            'zonetype': zone_type
                        })

                        index = i  
                        break 

            index += 1 

        zone_df = pd.DataFrame(zones)

        return zone_df



    # Dataframe of Zone pattern and types for daily interval
    zone_daily_df = zone_pattern(daily_df)

    # Dataframe of Zone pattern and types for 125min interval
    zone_125_df = zone_pattern(int125_df)

    # Dataframe of Zone pattern and types for 15min interval
    zone_15_df = zone_pattern(int15_df)

    # Function for determining zone violation
    def check_violation(zone_df, original_df):
        pattern_df = zone_df.copy()
        base_df = original_df.copy()

        pattern_df = pattern_df.iloc[::-1].reset_index(drop=True)
        pattern_df['violation'] = 'no'

        for i in range(len(pattern_df)):
            zone_datetime = pattern_df.at[i, 'leg-in-datetime']

            filtered_base_df = base_df[base_df['datetime'] > zone_datetime]
            filtered_base_df = filtered_base_df.reset_index(drop=True)

            if pattern_df.at[i, 'zonetype'] == 'demand':
                violation = filtered_base_df['intl'].min() < pattern_df.at[i, 'zonelow']
            elif pattern_df.at[i, 'zonetype'] == 'supply':
                violation = (filtered_base_df['inth'] > pattern_df.at[i, 'zonehigh']).any()

            if violation:
                pattern_df.at[i, 'violation'] = 'yes'

        return pattern_df


    # Dataframe of zone-violation for daily interval
    daily_violation = check_violation(zone_daily_df, daily_df)

    # Dataframe of zone-violation for 125min interval
    int125_violation = check_violation(zone_125_df, int125_df)

    # Dataframe of zone-violation for 15min interval
    int15_violation = check_violation(zone_15_df, int15_df)

    # Function for determining how many times any zone has been tested(pierced)
    def check_test(violation_df, original_df):
        pattern_df = violation_df.copy()
        base_df = original_df.copy()

        pattern_df['test'] = 0
        numeric_columns2 = ['zonehigh', 'zonelow', 'test', 'basecount']
        pattern_df[numeric_columns2] = pattern_df[numeric_columns2].apply(pd.to_numeric, errors='coerce')


        for i in range(len(pattern_df)):
            zone_datetime = pattern_df.at[i, 'leg-in-datetime']
            offset = pattern_df.at[i, 'basecount'] + 1

            filtered_base_df = base_df[base_df['datetime'] > zone_datetime]
            filtered_base_df = filtered_base_df.reset_index(drop=True)
            filtered_base_df = filtered_base_df.iloc[offset:]
            filtered_base_df = filtered_base_df.reset_index(drop=True)

            numeric_columns1 = ['into', 'inth', 'intl', 'intc']
            filtered_base_df[numeric_columns1] = filtered_base_df[numeric_columns1].apply(pd.to_numeric, errors='coerce')
            test_datetime = pd.to_datetime('1970-01-01 00:00:00')

            if pattern_df.at[i, 'zonetype'] == 'demand':
                count = 0
                for j in range(len(filtered_base_df)):
                    if pattern_df.at[i, 'zonelow'] <= filtered_base_df.at[j,'intl'] < pattern_df.at[i, 'zonehigh']:
                        count += 1
                        test_datetime = filtered_base_df.at[j,'datetime']
                pattern_df.at[i,'test_datetime'] = test_datetime
                pattern_df.at[i, 'test'] = count
            elif pattern_df.at[i, 'zonetype'] == 'supply':
                count = 0
                for j in range(len(filtered_base_df)):
                    if pattern_df.at[i, 'zonelow'] < filtered_base_df.at[j,'inth'] <= pattern_df.at[i, 'zonehigh']:
                        count += 1
                        test_datetime = filtered_base_df.at[j,'datetime']
                pattern_df.at[i, 'test'] = count
                pattern_df.at[i,'test_datetime'] = test_datetime

        return pattern_df
        

    # Dataframe of zone-test-count for daily interval
    daily_test_df = check_test(daily_violation, daily_df)

    # Dataframe of zone-test-count for 125min interval
    int125_test_df = check_test(int125_violation, int125_df)

    # Dataframe of zone-test-count for 15min interval
    int15_test_df = check_test(int15_violation, int15_df)

    # Function for determining trend(Up/Sideways/Down) for any interval
    def check_trend(df):
        sam = df[df['violation'] == 'yes']
        sam = sam.reset_index(drop=True)

        trend = 'sideways'

        if not sam.empty:
            if sam.at[0, 'zonetype'] == 'supply':
                if sam.at[1, 'zonetype'] == 'supply':
                    trend = 'up'
                elif sam.at[1, 'zonetype'] == 'demand':
                    trend = 'sideways'
            elif sam.at[0, 'zonetype'] == 'demand':
                if sam.at[1, 'zonetype'] == 'demand':
                    trend = 'down'
                elif sam.at[1, 'zonetype'] == 'supply':
                    trend = 'sideways'
            else:
                trend = 'sideways'

        return trend

    # Trend for daily interval
    daily_trend = check_trend(daily_violation)
    # Trend for 125min interval
    int125_trend = check_trend(int125_violation)
    # Trend for 15min interval
    int15_trend = check_trend(int15_violation)
                

    # Function for calculating risk and reward percentages for given(15min) interval
    def check_risk_reward(dataframe):

        i = 0
        while(i < len(dataframe)-1):
            if dataframe.at[i, 'zonetype'] == 'demand':
                if dataframe.at[i+1, 'zonetype'] == 'supply':
                    high = dataframe.at[i+1, 'zonelow']
                    mid = dataframe.at[i, 'zonehigh']
                    low = dataframe.at[i, 'zonelow']
                    reward = abs(high - mid)/abs(high-low)
                    risk = abs(mid - low)/abs(high-low)
            
            if dataframe.at[i, 'zonetype'] == 'supply':
                if dataframe.at[i+1, 'zonetype'] == 'demand':
                    high = dataframe.at[i, 'zonelow']
                    mid = dataframe.at[i+1, 'zonehigh']
                    low = dataframe.at[i+1, 'zonelow']
                    reward = abs(high - mid)/abs(high-low)
                    risk = abs(mid - low)/abs(high-low)
                    
            i += 1

        return [round(reward,2),round(risk,2)]
                

    # Store the Risk/Reward percentages in the list for 15min interval
    list = check_risk_reward(zone_15_df)


    # Function for determining Location value of the latest price(HOC/Equillibrium/LOC)
    def location_curve(df, original_df):
        sam = df[df['violation'] == 'no']
        sam = sam.reset_index(drop=True)
        curr_price = original_df.at[len(original_df)-1,'intc']
        top = 0
        bottom = 0

        i = 0
        while(i < len(sam)-1):
            if sam.at[i, 'zonetype'] == 'demand':
                if sam.at[i+1, 'zonetype'] == 'supply':
                    top = sam.at[i+1, 'zonehigh']
                    bottom = sam.at[i, 'zonelow']
            
            if sam.at[i, 'zonetype'] == 'supply':
                if sam.at[i+1, 'zonetype'] == 'demand':
                    top = sam.at[i, 'zonehigh']
                    bottom = sam.at[i+1, 'zonelow']
            i += 1
        
        loc_curve = 'NA'
        val = (top-bottom)/3
        if curr_price > bottom and curr_price < (bottom+val):
            loc_curve = 'LOC'
        elif curr_price < (bottom+(2*val)) and curr_price > (bottom+val):
            loc_curve = 'Equillibrium'
        elif curr_price > (bottom+(2*val)) and curr_price < top:
            loc_curve = 'HOC'

        return loc_curve
                

    # Calculating Location curve for daily interval
    curve = location_curve(daily_violation, daily_df)

    # Function for calculating the high and low values of candle with highest liquidity(volume)
    def check_liquidity(dataframe):
        liq_df = dataframe.copy()
        liq_df = liq_df.iloc[::-1].reset_index(drop=True)
        liq_df = liq_df.iloc[0:20].reset_index(drop=True)
        max_vol_index = liq_df['vol'].idxmax()
        liq_high = liq_df.at[max_vol_index, 'inth']
        liq_low = liq_df.at[max_vol_index, 'intl']
        return [liq_high, liq_low]
        
        
    # Store the values of high and low of candle with high liquidity
    liq = check_liquidity(daily_df)

    
    # Function for calculating signal type for given interval
    def check_signal_type(dataframe):
        output = ""
        if dataframe.at[0,'zonetype'] == "supply":
            if dataframe.at[0,'violation'] == "yes":
                output = "Near violated SZ"
            elif dataframe.at[0,'test'] > 0:
                output = "Near Pierced SZ"
            else:
                output = "Near SZ"
        elif dataframe.at[0,'zonetype'] == "demand":
            if dataframe.at[0,'violation'] == "yes":
                output = "Near violated DZ"
            elif dataframe.at[0,'test'] > 0:
                output = "Near Pierced DZ"
            else:
                output = "Near DZ"
        return output

    # Calculating signal type for 15min interval
    signal_type = check_signal_type(int15_test_df)

    
    
    # Function for determining if equal-high and equal-low is present for given interval
    def check_eq(test_df, original_df):
        temp_df = test_df[test_df['test'] == 0]
        temp_df = temp_df[temp_df['violation'] == 'no'].reset_index(drop=True)
        temp_df['eq'] = 'no'
        for k in range(len(temp_df)):
            piv_time = temp_df.iloc[k]['leg-out-datetime']
            piv_zone = temp_df.iloc[k]['zonetype']
            some_df = original_df[original_df['datetime'] > piv_time].reset_index(drop=True)
            # pivot_df = pd.DataFrame(columns=['datetime','into','intc','inth','intl','vol','diff'])
            pivot = []
            if piv_zone == 'supply':
                for i in range(3, len(some_df) - 3):
                    if (some_df.iloc[i]['inth'] > some_df.iloc[i - 3:i]['inth'].max()) and (some_df.iloc[i]['inth'] > some_df.iloc[i + 1:i + 4]['inth'].max()):
                        output = {
                        'datetime' : some_df.iloc[i]['datetime'],
                        'into' : some_df.iloc[i]['into'],
                        'intc' : some_df.iloc[i]['intc'],
                        'inth' : some_df.iloc[i]['inth'],
                        'intl' : some_df.iloc[i]['intl'],
                        'vol' : some_df.iloc[i]['vol'],
                        'diff' : abs(some_df.iloc[i]['into'] - some_df.iloc[i]['intc'])
                        }
                        pivot.append(output)
                if not pivot:  
                    continue
                result = pd.DataFrame(pivot)
                # print(result)
                result['eqh_count'] = 0
                difference = result['diff'].mean()
                for i in range(len(result)):
                    # difference = result.at[i,'diff']
                    curr_inth = result.at[i,'inth']
                    top = curr_inth + 0.1*difference
                    bottom = curr_inth - 0.1*difference
                    count = 0
                    for j in range(i,len(result)):
                        if bottom < result.at[j,'inth'] < top:
                            count += 1
                    result.at[i,'eqh_count'] = count
                if (result['eqh_count'] > 2).any():
                    temp_df.at[k,'eq'] = 'yes'
            
            elif piv_zone == 'demand':
                for i in range(3, len(some_df) - 3):
                    if (some_df.iloc[i]['intl'] < some_df.iloc[i - 3:i]['intl'].min()) and (some_df.iloc[i]['intl'] < some_df.iloc[i + 1:i + 4]['intl'].min()):
                        output = {
                        'datetime' : some_df.iloc[i]['datetime'],
                        'into' : some_df.iloc[i]['into'],
                        'intc' : some_df.iloc[i]['intc'],
                        'inth' : some_df.iloc[i]['inth'],
                        'intl' : some_df.iloc[i]['intl'],
                        'vol' : some_df.iloc[i]['vol'],
                        'diff' : abs(some_df.iloc[i]['into'] - some_df.iloc[i]['intc'])
                        }
                        pivot.append(output)
                if not pivot:  
                    continue
                result = pd.DataFrame(pivot)
                # print(result)
                result['eqh_count'] = 0
                difference = result['diff'].mean()
                for i in range(len(result)):
                    # difference = result.at[i,'diff']
                    curr_inth = result.at[i,'intl']
                    top = curr_inth + 0.1*difference
                    bottom = curr_inth - 0.1*difference
                    count = 0
                    for j in range(i,len(result)):
                        if bottom < result.at[j,'intl'] < top:
                            count += 1
                    result.at[i,'eqh_count'] = count
                if (result['eqh_count'] > 2).any():
                    temp_df.at[k,'eq'] = 'yes'

        return temp_df





    # Determining equal-high and equal-low presence for 15min interval
    int15_eq_df = check_eq(int15_test_df, int15_df)

    # Determining equal-high and equal-low presence for 125min interval
    int125_eq_df = check_eq(int125_test_df, int125_df)


    # Determining equal-high and equal-low presence for daily interval
    daily_eq_df = check_eq(daily_test_df, daily_df)

    # Code-Block for calculating equal-high/equal-low for 15min interval
    #~~~~~~
    int15_eq = 0
    int15_eqhigh = 0
    int15_eqlow = 0

    if (int15_eq_df['eq'] == 'yes').any():
        i = 0
        for i in range(len(int15_eq_df)):
            if int15_eq_df.at[i,'eq'] == 'yes':
                int15_eqhigh = int15_eq_df.at[i,'zonehigh']
                int15_eqlow = int15_eq_df.at[i,'zonelow']
                int15_eq = 1
                break
    #~~~~~~
            

    # Code-Block for calculating equal-high/equal-low for 125min interval
    #~~~~~~
    int125_eq = 0
    int125_eqhigh = 0
    int125_eqlow = 0

    if (int125_eq_df['eq'] == 'yes').any():
        i = 0
        for i in range(len(int15_eq_df)):
            if int125_eq_df.at[i,'eq'] == 'yes':
                int125_eqhigh = int15_eq_df.at[i,'zonehigh']
                int125_eqlow = int15_eq_df.at[i,'zonelow']
                int125_eq = 1
                break
    #~~~~~~
            

    # Code-Block for calculating equal-high/equal-low for daily interval
    #~~~~~~
    daily_eq = 0
    daily_eqhigh = 0
    daily_eqlow = 0

    if (daily_eq_df['eq'] == 'yes').any():
        i = 0
        for i in range(len(daily_eq_df)):
            if daily_eq_df.at[i,'eq'] == 'yes':
                daily_eqhigh = daily_eq_df.at[i,'zonehigh']
                daily_eqlow = daily_eq_df.at[i,'zonelow']
                daily_eq = 1
                break
    #~~~~~~
            


    # Function for determining Gap presence, Gap-exit, Gap-entry for given interval
    def check_gap(test_df, original_df):
        c1_time = test_df.iloc[0]['leg-out-datetime']
        ztype = test_df.iloc[0]['zonetype']
        test = test_df.iloc[0]['test_datetime']
        
        some_df = original_df[original_df["datetime"] >= c1_time]
        some_df = some_df.reset_index(drop=True)
        val = some_df['datetime'].count()
        gap_exit = 0
        gap = 0
        if val >= 3:
            if ztype == 'supply':
                gap_exit = some_df.at[2, 'inth'] - some_df.at[0, 'intl']
            elif ztype == 'demand':
                gap_exit = some_df.at[2, 'intl'] - some_df.at[0, 'inth']
        if gap_exit > 0:
            gap = 1
        gap_exit = round(gap_exit, 2)


        gap_entry = 0
        if test != pd.to_datetime('1970-01-01 00:00:00'):
            temp_df = int15_df[int15_df["datetime"] <= test]    
            temp_df = temp_df.iloc[::-1].reset_index(drop=True)
            if val >= 3:
                if ztype == 'supply':
                    gap_entry = temp_df.at[0, 'intl'] - temp_df.at[2, 'inth']
                    # gap_entry = (temp_df.at[0, 'intl'] - temp_df.at[2, 'inth'])/temp_df.at[2, 'inth']
                elif ztype == 'demand':
                    gap_entry = temp_df.at[2, 'intl'] - temp_df.at[0, 'inth']
                    # gap_entry = (temp_df.at[2, 'intl'] - temp_df.at[0, 'inth'])/temp_df.at[2, 'intl']
        gap_entry = round(gap_entry, 2)

        return [gap, gap_exit, gap_entry]

    # Calculating Gap presence, Gap-exit, Gap-entry for 15min interval
    checkgap = check_gap(int15_test_df, int15_df)



    def check_target(zone_df):
        target = 0
        stop_loss = 0
        if not zone_df.empty:
            if zone_df.at[0,'zonetype'] == 'supply':
                stop_loss = zone_df.at[0,'zonehigh']
                for i in range(len(zone_df)):
                    if zone_df.at[i,'zonetype'] == 'demand':
                        target = zone_df.at[i,'zonehigh']
                        break
            elif zone_df.at[0,'zonetype'] == 'demand':
                stop_loss = zone_df.at[0,'zonelow']
                for i in range(len(zone_df)):
                    if zone_df.at[i,'zonetype'] == 'supply':
                        target = zone_df.at[i,'zonelow']
                        break
                    
        return [target, stop_loss]

    Target = check_target(zone_daily_df)[0]
    Stop_Loss = check_target(zone_daily_df)[1]



    # List of columns present in final Dataframe
    columns = [
        'Script', 'Date', 'Ltp_Price', 'High_tf_trend', 'Intermediate_tf_trend',
        'Low_tf_trend', 'Signal_type', 'Nearest_zone', 'Pattern', 'Test', 'Location_Curve',
        'Leg_in_candle', 'Base', 'Leg_out_candle', 'Zone_high',
        'Zone_low', 'Reward', 'Risk', 'Eql_Eqh_int15', 'Eql_Eqh_int125', 'Eql_Eqh_daily', 'int15_eq_high', 'int125_eq_high', 'daily_eq_high', 'int15_eq_low', 'int125_eq_low', 'daily_eq_low', 'Liq_high', 'Liq_low', 'Gap', 'Gap_exit', 'Gap_entry', 'Entry_Price', 'Target', 'Stop_Loss', 'Current_Panel'
    ]

    # Calculating Remaining Values required for Final Dataframe
    Date = df.iloc[-1]['time']
    Ltp = daily_df.iloc[-1]['intc']
    Nearest_zone = daily_test_df.iloc[0]['zonetype']
    Pattern = daily_test_df.iloc[0]['zonepattern']
    Test = daily_test_df.iloc[0]['test']
    Legin = daily_test_df.iloc[0]['leg-in-datetime']
    Basecount = daily_test_df.iloc[0]['basecount']
    Legout = daily_test_df.iloc[0]['leg-out-datetime']
    Zonehigh = daily_test_df.iloc[0]['zonehigh']
    Zonelow = daily_test_df.iloc[0]['zonelow']
    Liq_high = liq[0]
    Liq_low = liq[1]
    Gap = checkgap[0]
    Gap_Exit = checkgap[1]
    Gap_Entry = checkgap[2]
    Reward = list[0]
    Risk = list[1]



    # Plotting the values as a row for Final Dataframe

    new_row = [
        TradingSymbol, Date, Ltp, daily_trend, int125_trend, int15_trend, signal_type, Nearest_zone, Pattern, Test, curve, Legin, Basecount, Legout, Zonehigh, Zonelow, Reward, Risk, int15_eq, int125_eq, daily_eq, int15_eqhigh, int125_eqhigh, daily_eqhigh, int15_eqlow, int125_eqlow, daily_eqlow, Liq_high, Liq_low, Gap, Gap_Exit, Gap_Entry, Ltp, Target, Stop_Loss, 0
    ]

    # Making a dctionary format out of the columns and row-values
    data_dict = dict(zip(columns, new_row))

    return data_dict



# Dataframe containing the Tokens and Trading-Symbols
fno_symbols=pd.read_csv('https://api.shoonya.com/NSE_symbols.txt.zip')

# List of all the Trading Symbols required
fnostocks = [
'RECLTD-EQ',
'HDFCBANK-EQ',
'AXISBANK-EQ',
'RELIANCE-EQ',
'INFY-EQ',
'PFC-EQ',
'ICICIBANK-EQ',
'SBIN-EQ',
'BANKBARODA-EQ',
'TCS-EQ',
'ADANIPORTS-EQ',
'TATAPOWER-EQ',
'VEDL-EQ',
'PIIND-EQ',
'ULTRACEMCO-EQ',
'GMRINFRA-EQ',
'DIXON-EQ',
'MARUTI-EQ',
'NTPC-EQ',
'NMDC-EQ',
'HINDPETRO-EQ',
'TVSMOTOR-EQ',
'KOTAKBANK-EQ',
'BANDHANBNK-EQ',
'ADANIENT-EQ',
'BAJFINANCE-EQ',
'PNB-EQ',
'ITC-EQ',
'M&M-EQ',
'IDFCFIRSTB-EQ',
'TATAMOTORS-EQ',
'HAL-EQ',
'RBLBANK-EQ',
'HEROMOTOCO-EQ',
'BPCL-EQ',
'HDFCLIFE-EQ',
'AMBUJACEM-EQ',
'CANBK-EQ',
'LT-EQ',
'INDIGO-EQ',
'BIOCON-EQ',
'ASIANPAINT-EQ',
'COALINDIA-EQ',
'INDUSINDBK-EQ',
'WIPRO-EQ',
'TATASTEEL-EQ',
'ASHOKLEY-EQ',
'GAIL-EQ',
'BHEL-EQ',
'POWERGRID-EQ',
'IDEA-EQ',
'DRREDDY-EQ',
'EICHERMOT-EQ',
'CONCOR-EQ',
'IRCTC-EQ',
'JINDALSTEL-EQ',
'BAJAJ-AUTO-EQ',
'TECHM-EQ',
'DLF-EQ',
'PIDILITIND-EQ',
'HINDALCO-EQ',
'JSWSTEEL-EQ',
'AUROPHARMA-EQ',
'CHAMBLFERT-EQ',
'POLYCAB-EQ',
'APOLLOTYRE-EQ',
'TRENT-EQ',
'BHARTIARTL-EQ',
'BEL-EQ',
'AUBANK-EQ',
'HINDUNILVR-EQ',
'COFORGE-EQ',
'FEDERALBNK-EQ',
'LAURUSLABS-EQ',
'BAJAJFINSV-EQ',
'ONGC-EQ',
'INDUSTOWER-EQ',
'HCLTECH-EQ',
'APOLLOHOSP-EQ',
'GLENMARK-EQ',
'PEL-EQ',
'SUNPHARMA-EQ',
'LTIM-EQ',
'CUMMINSIND-EQ',
'IOC-EQ',
'SYNGENE-EQ',
'ACC-EQ',
'VOLTAS-EQ',
'MUTHOOTFIN-EQ',
'MCDOWELL-N-EQ',
'UPL-EQ',
'CANFINHOME-EQ',
'MIDCPNIFTY-EQ',
'IEX-EQ',
'CIPLA-EQ',
'GODREJCP-EQ',
'TATACHEM-EQ',
'SBICARD-EQ',
'TATACOMM-EQ',
'GRASIM-EQ',
'M&MFIN-EQ',
'SRF-EQ',
'SIEMENS-EQ',
'IGL-EQ',
'COLPAL-EQ',
'NAUKRI-EQ',
'HAVELLS-EQ',
'L&TFH-EQ',
'ZYDUSLIFE-EQ',
'INDIAMART-EQ',
'BHARATFORG-EQ',
'ABFRL-EQ',
'TITAN-EQ',
'ALKEM-EQ',
'CHOLAFIN-EQ',
'DIVISLAB-EQ',
'LUPIN-EQ',
'PERSISTENT-EQ',
'BERGEPAINT-EQ',
'LICHSGFIN-EQ',
'AARTIIND-EQ',
'SHRIRAMFIN-EQ',
'MCX-EQ',
'MPHASIS-EQ',
'BSOFT-EQ',
'GODREJPROP-EQ',
'RAMCOCEM-EQ',
'DEEPAKNTR-EQ',
'BALKRISIND-EQ',
'ABB-EQ',
'HDFCAMC-EQ',
'ICICIPRULI-EQ',
'METROPOLIS-EQ',
'MOTHERSON-EQ',
'PETRONET-EQ',
'ESCORTS-EQ',
'GNFC-EQ',
'GRANULES-EQ',
'MARICO-EQ',
'MRF-EQ',
'SBILIFE-EQ',
'OBEROIRLTY-EQ',
'MANAPPURAM-EQ',
'DALBHARAT-EQ',
'CUB-EQ',
'INDHOTEL-EQ',
'SHREECEM-EQ',
'LALPATHLAB-EQ',
'IPCALAB-EQ',
'PAGEIND-EQ',
'EXIDEIND-EQ',
'FINNIFTY-EQ',
'TATACONSUM-EQ',
'ASTRAL-EQ',
'UBL-EQ',
'ZEEL-EQ',
'GUJGASLTD-EQ',
'ABCAPITAL-EQ',
'DABUR-EQ',
'IDFC-EQ',
'SAIL-EQ',
'LTTS-EQ',
'ICICIGI-EQ',
'OFSS-EQ',
'NESTLEIND-EQ',
'JUBLFOOD-EQ',
'CROMPTON-EQ',
'MGL-EQ',
'NAVINFLUOR-EQ',
'BATAINDIA-EQ',
'PVRINOX-EQ',
'BRITANNIA-EQ',
'TORNTPHARM-EQ',
'MFSL-EQ',
'SUNTV-EQ',
'ATUL-EQ',
'COROMANDEL-EQ',
'JKCEMENT-EQ',
'BOSCHLTD-EQ',
'ABBOTINDIA-EQ',
'HINDCOPPER-EQ',
'INDIACEM-EQ',
'IBULHSGFIN-EQ',
'NATIONALUM-EQ',
'BALRAMCHIN-EQ',
'DELTACORP-EQ'
]


# Dataframe of the Tokens and Trading-Symbols from the list
filtered_df = fno_symbols[fno_symbols['TradingSymbol'].isin(fnostocks)]
result = filtered_df[['Token', 'TradingSymbol']].reset_index(drop=True)
result['Token'] = result['Token'].astype(str)

# Making a list (it will contain all the rows for each Trading_symbol in the list in form of dictionary)
stocks = []

# Loop through all the Trading-Symbols in the fnostocks list and get the new row for Final Dataframe
length = 3 # No. of rows to be Showcased on the web-app (For now it shows the data for first three Trading-Symbols)
for i in range(length):
    output = main(result.at[i,'Token'],result.at[i,'TradingSymbol'])
    stocks.append(output)

# Converting all the data collected to a Dataframe
stocks_df = pd.DataFrame(stocks)
print(stocks_df)

# Streamlit code for page layout
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
        body {
            padding: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Code for showcasing the Final Dataframe on the web-app
st.title("FNO STOCKS DATA")
st.dataframe(stocks_df)