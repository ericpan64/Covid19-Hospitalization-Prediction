
# TODO: download the data 

# get the abstract
wget https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-11-12/metadata.csv
pip install csvtool
csvtool -c 9 metadata.csv > abstracts.txt
csvtool -c 4 metadata.csv > titles.txt

# get the packages
pip install scispacy
pip install spacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_ner_bc5cdr_md-0.3.0.tar.gz
