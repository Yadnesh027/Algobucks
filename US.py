import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date
import yfinance as yf

def main(TradingSymbol):
    # Collecting the Data     
    msft = yf.Ticker(TradingSymbol)

    
    # Get historical market data for 15min interval
    int15_df = msft.history(period="3d",interval='15m')

    # Get historical market data for 5min interval
    int130_df = msft.history(period="10d",interval='5m')
    
    # Get historical market data for daily interval
    daily_df = msft.history(period="200d",interval='1d')

 
    def update_column(df):
    
        df['datetime'] = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S%z')
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S%z').dt.strftime('%Y-%m-%d %H:%M:%S')
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        df = df.reset_index(drop=True)
        df.rename(columns={'Open': 'into'}, inplace=True)
        df.rename(columns={'Close': 'intc'}, inplace=True)
        df.rename(columns={'High': 'inth'}, inplace=True)
        df.rename(columns={'Low': 'intl'}, inplace=True)
        df.rename(columns={'Volume': 'vol'}, inplace=True)
        df = df[['datetime', 'into', 'inth', 'intl', 'intc', 'vol']]

        return df

    
    
    int15_df = update_column(int15_df)
    int130_df = update_column(int130_df)
    daily_df = update_column(daily_df)


    def interval_buckets(df, interval):
        n = int(interval/5)
        new_data = []

        for i in range(0, len(df), n):
            temp_df = df.iloc[i:i+n]


            if len(temp_df) >= 1:
                datetime = temp_df.iloc[0]['datetime']
                # time = temp_df.iloc[0]['datetime']
                into = temp_df.iloc[0]['into']
                intc = temp_df.iloc[-1]['intc']
                vol = temp_df['vol'].sum()

                inth = temp_df['inth'].max()
                intl = temp_df['intl'].min()

                new_data.append({'datetime': datetime, 'into': into, 'intc': intc, 'inth': inth, 'intl': intl, 'vol':vol})

        new_df = pd.DataFrame(new_data)

        return new_df


    int130_df = interval_buckets(int130_df,130)




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

    # Dataframe of Zone pattern and types for 130min interval
    zone_130_df = zone_pattern(int130_df)

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

    # Dataframe of zone-violation for 130min interval
    int130_violation = check_violation(zone_130_df, int130_df)

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

    # Dataframe of zone-test-count for 130min interval
    int130_test_df = check_test(int130_violation, int130_df)

    # Dataframe of zone-test-count for 15min interval
    int15_test_df = check_test(int15_violation, int15_df)

    # Function for determining trend(Up/Sideways/Down) for any interval
    def check_trend(df):
        sam = df[df['violation'] == 'yes']
        sam = sam.reset_index(drop=True)
        print(sam)

        trend = 'sideways'

        if len(sam) >= 2:
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

    # Trend for 130min interval
    int130_trend = check_trend(int130_violation)

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

    # Determining equal-high and equal-low presence for 130min interval
    int130_eq_df = check_eq(int130_test_df, int130_df)


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
            

    # Code-Block for calculating equal-high/equal-low for 130min interval
    #~~~~~~
    int130_eq = 0
    int130_eqhigh = 0
    int130_eqlow = 0

    if (int130_eq_df['eq'] == 'yes').any():
        i = 0
        for i in range(len(int15_eq_df)):
            if int130_eq_df.at[i,'eq'] == 'yes':
                int130_eqhigh = int15_eq_df.at[i,'zonehigh']
                int130_eqlow = int15_eq_df.at[i,'zonelow']
                int130_eq = 1
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

    # List of columns present in final Dataframe
    columns = [
        'Script', 'Date', 'Ltp_Price', 'High_tf_trend', 'Intermediate_tf_trend',
        'Low_tf_trend', 'Signal_type', 'Nearest_zone', 'Pattern', 'Test', 'Location_Curve',
        'Leg_in_candle', 'Base', 'Leg_out_candle', 'Zone_high',
        'Zone_low', 'Reward', 'Risk', 'Eql_Eqh_int15', 'Eql_Eqh_int130', 'Eql_Eqh_daily', 'int15_eq_high', 'int130_eq_high', 'daily_eq_high', 'int15_eq_low', 'int130_eq_low', 'daily_eq_low', 'Liq_high', 'Liq_low', 'Gap', 'Gap_exit', 'Gap_entry'
    ]

    # Calculating Remaining Values required for Final Dataframe
    Date = int15_df.iloc[-1]['datetime']
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
        TradingSymbol, Date, "NA", daily_trend, int130_trend, int15_trend, signal_type, Nearest_zone, Pattern, Test, curve, Legin, Basecount, Legout, Zonehigh, Zonelow, Reward, Risk, int15_eq, int130_eq, daily_eq, int15_eqhigh, int130_eqhigh, daily_eqhigh, int15_eqlow, int130_eqlow, daily_eqlow, Liq_high, Liq_low, Gap, Gap_Exit, Gap_Entry
    ]

    # Making a dctionary format out of the columns and row-values
    data_dict = dict(zip(columns, new_row))

    return data_dict





fnostocks = [
    "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "BRK-A", "TSLA", "LLY", "JPM", "V", "TSM",
    "AVGO", "NVO", "NONOF", "UNH", "WMT", "MA", "XOM", "JNJ", "LVMHF", "LVMUY", "TCTZF", "PG", "HD", "TCEHY", "LTMAY",
    "JPM-PC", "JPM-PD", "BAC", "BAC-PK", "NSRGY", "NSRGF", "BML-PG", "COST", "MRK", "BAC-PE", "ORCL", "BML-PH", "BAC-PL",
    "BML-PL", "ABBV", "ASMLF", "ASML", "CVX", "ADBE", "TM", "TOYOF", "CRM", "LRLCF", "KO", "LRLCY", "BML-PJ", "BAC-PB",
    "RHHBF", "RHHBY", "RHHVF", "AMD", "FMX", "PEP", "IDCBY", "ACN", "NVS", "NVSEF", "IDCBF", "AZNCF", "NFLX", "AZN",
    "MCD", "HESAF", "HESAY", "TMO", "SHEL", "RYDAF", "CSCO", "INTC", "LIN", "ABT", "PDD", "WFC-PY", "MTSUY", "TMUS",
    "SAPGF", "WFC-PR", "SAP", "WFC-PL", "BABAF", "BABA", "ACGBY", "PCCYF", "ACGBF", "CMCSA", "WFC", "INTU", "C-PJ",
    "DHR", "HDB", "DIS", "BHPLF", "AMGN", "PROSY", "VZ", "PFE", "BHP", "TTFNF", "NKE", "PROSF", "MS", "QCOM", "TTE",
    "HSBC", "HBCYF", "IBM", "TXN", "NOW", "CICHY", "CICHF", "PM", "CAT", "UNP", "BX", "GS", "GE", "SIEGY", "SMAWF",
    "RTNTF", "BACHY", "RY", "SPGI", "CMWAY", "UPS", "IDEXF", "BACHF", "SNYNF", "IDEXY", "CHDRY", "AXP", "COP", "CBAUF",
    "HON", "C", "BA", "SNY", "BUD", "UBER", "BUDFF", "EADSF", "EADSY", "CHDRF", "ISRG", "LOW", "AMAT", "WFC-PC", "NEE",
    "UNLYF", "RTX", "SONY", "DTEGF", "UL", "DTEGY", "BKNG", "PLD", "SNEJF", "BLK", "SCHW", "SYK", "T", "MDT", "LMT",
    "RIO", "RTPPF", "KYCCF", "VRTX", "ELV", "TD", "SBGSF", "SBGSY", "TJX", "DE", "GILD", "ALIZF", "PBR", "MUFG",
    "NPPXF", "NTTYY", "SBUX", "SHOP", "ALIZY", "PANW", "PBR-A", "BMY", "REGN", "MBFJF", "BPAQF", "AIQUF", "AIQUY",
    "RLLCF", "BP", "LRCX", "MDLZ", "PGR", "CVS", "CIHKY", "AMT", "ADP", "ETN", "MMC", "UBS", "CMXHF", "CSLLY", "ADI",
    "CB", "AAGIY", "ESLOF", "ESLOY", "ZTS", "MU", "CSUAY", "PNGAY", "EQNR", "AAIGF", "STOHF", "CI", "PIAIF", "CUAEF",
    "ABNB", "BSX", "USB-PH", "BNPQF", "SNPMF", "TOELF", "BNPQY", "IBN", "GS-PK", "MELI", "TOELY", "GSK", "GLAXF",
    "MS-PK", "FI", "FRCOY", "CNI", "CILJF", "GS-PA", "DGEAF", "INFY", "MS-PI", "ABLZF", "DEO", "MS-PF", "RLXXF",
    "SHECF", "IBDSF", "BYDDF", "BYDDY", "ATLKY", "ABBNY", "IBDRY", "SHECY", "ENB", "ANET", "FRCOF", "MS-PE", "PBCRY",
    "GS-PD", "SO", "RELX", "SHW", "SAFRF", "ZFSVF", "CFRHF", "EQIX", "MS-PA", "SAFRY", "ITW", "KLAC", "DUK", "PBCRF",
    "CFRUY", "HCA", "EBBNF", "SNPS", "ENLAY", "ZURVY", "ESOCF", "ATLCY", "ATLPF", "WDAY", "AXAHF", "USB-PP", "WMMVY",
    "WMMVF", "MO", "WM", "CDNS", "VCISF", "VCISY", "AXAHY", "KKR", "ICE", "MBGAF", "MBGYY", "NOC", "CP", "BYMOF",
    "ARM", "SLB", "CME", "CNQ", "KDDIY", "BMWYY", "HTHIF", "GLCNF", "HTHIY", "GLNCY", "BAMXF", "BTAFF", "KDDIF",
    "GD", "MCO", "MSBHF", "MKKGY", "CSX", "RCRRF", "BDX", "IVSXF", "IVSBF", "CRWD", "EOG", "BTI", "STLA", "SMFNF", "MKGAF", "ITUB", "SBKFF", "CHGCY", "RCRUY", "CL", "BCDRF", "TRI", "MAR", "ITOCF",
    "SAN", "DASTY", "SMFG", "VWAGY", "PYPL", "USB", "SMMNY", "SFTBF", "NTDOF", "NTDOY", "ITOCY", "SWGAY", "SFTBY", "TGT",
    "SEMHF", "DASTF", "MCK", "UNCFF", "SCCO", "VWAPY", "LNSTY", "VLKPF", "VALE", "UNCRY", "DBSDY", "RACE", "OLCLY",
    "SNOW", "DBSDF", "LDNXF", "VLKAF", "DSKYF", "CMG", "SFBQF", "MITSF", "FDX", "DUK-PA", "SOBKY", "CTA-PB", "MNST",
    "OLCLF", "TEAM", "BN", "MPNGF", "MITSY", "LULU", "BMO", "MPNGY", "CTAS", "AON", "DSNKY", "MPC", "NETTF", "MMM",
    "PNC", "NTES"
]





# Making a list (it will contain all the rows for each Trading_symbol in the list in form of dictionary)
stocks = []

# Loop through all the Trading-Symbols in the fnostocks list and get the new row for Final Dataframe
length = 3
 # No. of rows to be Showcased on the web-app (For now it shows the data for first three Trading-Symbols)
for i in range(length):
    output = main(fnostocks[i])
    stocks.append(output)

# Converting all the data collected to a Dataframe
stocks_df = pd.DataFrame(stocks)

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
