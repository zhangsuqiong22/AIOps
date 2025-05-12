import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import argparse
from joblib import dump
import os


# 时间格式和常量
time_format = '%H:%M:%S'
minute_in_hour = 60
secs_in_minute = 60

# 时间戳的周期性编码
def pos_sin_sec(time_str):
    seconds = datetime.strptime(time_str, time_format).second
    return np.sin(2 * np.pi * seconds / secs_in_minute)

def pos_cos_sec(time_str):
    seconds = datetime.strptime(time_str, time_format).second
    return np.cos(2 * np.pi * seconds / secs_in_minute)

def pos_sin_min(time_str):
    minutes = datetime.strptime(time_str, time_format).minute
    return np.sin(2 * np.pi * minutes / minute_in_hour)

def pos_cos_min(time_str):
    minutes = datetime.strptime(time_str, time_format).minute
    return np.cos(2 * np.pi * minutes / minute_in_hour)

# 数据处理函数
def process_data(source):
    # 读取数据集
    df = pd.read_csv(source)

    # 添加时间的周期性编码
    df['sin_sec'] = df['timestamp'].apply(pos_sin_sec)
    df['cos_sec'] = df['timestamp'].apply(pos_cos_sec)
    df['sin_min'] = df['timestamp'].apply(pos_sin_min)
    df['cos_min'] = df['timestamp'].apply(pos_cos_min)


    memory_features = [col for col in df.columns if col.startswith('sys-mem')]
    disk_byte_features = [col for col in df.columns if col in ['disk-bytes-read', 'disk-bytes-written']]
    rate_features = [col for col in df.columns if any(x in col for x in ['rate', 'disk-io-read', 'disk-io-write'])]
    cpu_features = [col for col in df.columns if col.startswith('cpu')]
    load_features = [col for col in df.columns if col.startswith('load')]
    
    transform_methods = {}
    
    for col in memory_features + disk_byte_features:
        if col in df.columns:
            if (df[col] >= 0).all():
                df[col] = np.log1p(df[col])
                transform_methods[col] = 'log1p'

    for col in rate_features:
        if col in df.columns:
            if (df[col] >= 0).all():
                df[col] = np.sqrt(df[col])
                transform_methods[col] = 'sqrt'
    

    for col in load_features:
        if col in df.columns:
            if df[col].mean() > 1.0:
                df[col] = np.log1p(df[col])
                transform_methods[col] = 'log1p'
            else:
                transform_methods[col] = 'none'
    

    for col in df.columns:
        if col not in transform_methods and col != 'timestamp':
            transform_methods[col] = 'none'
    
    import json
    with open('transform_methods.json', 'w') as f:
        json.dump(transform_methods, f, indent=4)
    
    with open('feature_transformed_only.txt', 'w') as f:
        f.write('Data has been feature-transformed but not normalized.\n')
        f.write('The following transformations were applied:\n')
        for col, method in transform_methods.items():
            f.write(f'- {col}: {method}\n')
    
    return df

# 主函数
if __name__ == '__main__':
    os.makedirs('scalers', exist_ok=True)

    parser = argparse.ArgumentParser(description='Preprocess the data')
    parser.add_argument('--input', type=str, help='The input file path')
    parser.add_argument('--output', type=str, help='The output file path')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        print('Please provide the input and output file path')
        exit(1)

    # 处理数据并保存
    processed_data = process_data(args.input)
    processed_data.to_csv(args.output, index=False)