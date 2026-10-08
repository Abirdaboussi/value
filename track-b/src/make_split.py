import pandas as pd

df = pd.read_csv('data/processed/manifest.csv')
ok_df = df[df['ok'] == True]  # adjust column name if different

# set your chosen dev ids manually here
dev_ids = ['gao-20-215sp', 'gao-25-107335']  # replace with your picks

dev_df = ok_df[ok_df['doc_id'].isin(dev_ids)]
test_df = ok_df[~ok_df['doc_id'].isin(dev_ids)]

with open('data/splits/dev.txt', 'w') as f:
    f.write('\n'.join(dev_df['doc_id']))

with open('data/splits/test.txt', 'w') as f:
    f.write('\n'.join(test_df['doc_id']))

print(f"dev: {len(dev_df)} docs, test: {len(test_df)} docs")