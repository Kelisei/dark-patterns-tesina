import os
path = r'C:\Users\frank\Downloads\Tesina\dark-patterns-tesina\research\datasets\unified_dataset.csv'
with open(path, 'r', encoding='utf-8') as f:
    for line in f.readlines()[-5:]:
        print(repr(line))
