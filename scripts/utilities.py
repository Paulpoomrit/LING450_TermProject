import re
import csv
import pandas as pd
from feature_extractor import extract_features

def clean_and_convert_to_tsv(csv_path: str, dest_path: str) -> None:

    with open(csv_path) as f:
        content = f.read()
        content_new = re.sub(pattern=r",AI,",
                             repl=r"\tAI\t",
                             string=content)
        content_new = re.sub(pattern=r",H,",
                             repl=r"\tH\t",
                             string=content_new)

        content_new = re.sub(pattern=r",{2,}",
                             repl="",
                             string=content_new)

        content_new = re.sub(pattern=r"text,label",
                             repl="text\tlabel",
                             string=content_new)

        with open(dest_path, 'w') as tsv:
            tsv.write(content_new)


def extract_features_and_save(data_path: str, dest_path: str) -> None:

    df = pd.read_csv(data_path, sep='\t', index_col=False)
    # print(df['text'])

    feature_vectors = []
    count = 0
    for r in df.itertuples(index=False):

        print(f'Extracting features for sentence: {count} / {df.size} ')
        feature_vectors.append(extract_features(r.text))
        count += 1

    # print(feature_vectors)
    df['vector'] = feature_vectors
    print(df)
    df.to_csv(dest_path, sep='\t')



#clean_and_convert_to_tsv('data/test_data/test_corrupt.csv', 'data/test_data/test_not_corrupt.tsv')
extract_features_and_save('data/test_data/test_not_corrupt.tsv', 'data/test_data/w_features.tsv')