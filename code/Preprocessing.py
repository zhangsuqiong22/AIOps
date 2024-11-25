import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
from icecream import ic
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import argparse


minute_in_hour = 60
secs_in_minute = 60
ms_in_second = 1000
hours_in_day = 24

def pos_sin_ms(time_obj):
    microseconds = time_obj.microsecond
    sin_ms = np.sin(2 * np.pi * microseconds / ms_in_second)
    return sin_ms

def pos_cos_ms(time_obj):
    microseconds = time_obj.microsecond
    cos_ms = np.cos(2 * np.pi * microseconds / ms_in_second)
    return cos_ms

def pos_sin_sec(time_obj):
    second = time_obj.second
    sin_sec = np.sin(2 * np.pi * second / secs_in_minute)
    return sin_sec

def pos_cos_sec(time_obj):
    second = time_obj.second
    cos_sec = np.cos(2 * np.pi * second / secs_in_minute)
    return cos_sec

def pos_sin_min(time_obj):
    minute = time_obj.minute
    sin_min = np.sin(2 * np.pi * minute / minute_in_hour)
    return sin_min

def pos_cos_min(time_obj):
    minute = time_obj.minute
    cos_min = np.cos(2 * np.pi * minute / minute_in_hour)
    return cos_min

def pos_sin_hour(time_obj):
    hour = time_obj.hour
    sin_hour = np.sin(2 * np.pi * hour / hours_in_day)
    return sin_hour

def pos_cos_hour(time_obj):
    hour = time_obj.hour
    cos_hour = np.cos(2 * np.pi * hour / hours_in_day)
    return cos_hour

def process_data(source):
    df = pd.read_csv(source)

    # Rename 'timestamp' column to 'time'
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'time'}, inplace=True)

    if 'time' not in df.columns:
        raise KeyError("The 'time' column is missing from the input data")

    # Convert 'time' column from Unix timestamps to datetime objects
    df['time'] = pd.to_datetime(df['time'], unit='s')

    df['sin_ms'] = df['time'].apply(pos_sin_ms)
    df['cos_ms'] = df['time'].apply(pos_cos_ms)
    df['sin_sec'] = df['time'].apply(pos_sin_sec)
    df['cos_sec'] = df['time'].apply(pos_cos_sec)
    df['sin_min'] = df['time'].apply(pos_sin_min)
    df['cos_min'] = df['time'].apply(pos_cos_min)
    df['sin_hour'] = df['time'].apply(pos_sin_hour)
    df['cos_hour'] = df['time'].apply(pos_cos_hour)

    df['cpu_load'] = round(df['cgroupfs_user_cpu_usage_us'] * 100 / df['cgroupfs_cpu_usage_us'], 2)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--input', type=str, help='The input file path')
    parser.add_argument('--output', type=str, help='The output file path')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        print('Please provide the input and output file path')
        exit(1)

    train_dataset = process_data(args.input)
    train_dataset.to_csv(args.output, index=False)