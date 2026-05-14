import os
path = r'C:\Users\frank\Downloads\Tesina\dark-patterns-tesina\research\datasets\unified_dataset.csv'
with open(path, 'rb') as f:
    lines = f.readlines()
clean_lines = []
for line in lines[:-2]: # drop the two bad lines I just added
    clean_lines.append(line)
with open(path, 'wb') as f:
    f.writelines(clean_lines)
with open(path, 'a', encoding='utf-8') as f:
    f.write('shaming,\"Sí, me voy a hacer cargo de los gastos que surjan\",roomio_checkout\n')
    f.write('shaming,\"Yes, I will take care of any expenses that may be incurred\",roomio_checkout\n')
