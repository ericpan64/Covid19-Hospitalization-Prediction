This folder contains all you need to do the natural language processing part of the project.
The main idea is that whether we can extract some insights on published papers 
on covid 19 and guide us in choosing which features in the DREAM challenge dataset to focus on.

The published papers dataset, CORD-19, was created by Allen Institute for AI, available from [Semantic Scholar](https://www.semanticscholar.org/cord19) 
and [Kaggle](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).


## Folder content 
**script/**
- **`process.sh`** 
    - contains information on how to 
        - get started by downloading the main input (`metadata.csv`) from semantic scholar.
        - download `csvtool`, a CLI tool for extracting the columns (title, or abstracts) we need from `metadata.csv`
        - `spacy` - NLP Python package
        - `scispacy` - NLP Python package containing tools to work with spacy to load models specifically trained on biomedical text 
        and/or add other pipelines
        - download NLP model - `en_ner_bc5cdr_md`
    - `en_ner_bc5cdr_md` - NLP model trained to specifically recognize disease and chemical terms. This seems to be the most suiting model
    to used for this project in which we deal with health records, among [all other models](https://allenai.github.io/scispacy/)
    - `metadata.csv` : part of CORD-10 dataset. It contains information about each publication in CORD-19.  For the purpose of this project, we'll only
    be using titles (and maybe abstracts in the future) of the papers. So we don't need the full CORD-19 dataset.

- **`counts_concepts.ipynb`**
    - PySpark Jupyter notebook. The notebook contains information on how to run it
    - overall idea: the notebook
        - do a word count for each UMLS IDs from `umls_cui_in_titles.txt`
        - aggregate the counts for all UMLS IDS for each concept ID.
        - write the result in descending order of the aggregated counts for each DREAM challenge concept IDs.

- **`get_ids_from_abs.py`**
    - Python script for getting umls terms from abstract or titles. For each medical term identified, the model return 1-5 UMLS candidate IDs. We'll use all the IDs.
    - example run `python input/get_ids_from_abs.py --ab titles.txt --out output/umls_cui_in_titles.txt` with 16G of Memory, this would take about ~40 min
    
- **`get_ids_from_dict.py`**
    - Python script for getting umls terms from `data_dictionary.csv` work exactly the same as `get_ids_from_abs.py`
    - `python get_ids_from_dict.py --dict input/data_dictionary.csv --out output/ids_from_dict.txt`

**input/**
- **`data_dictionary.csv`**
    - from the Dream challenge. Map the concept ids from various tables in the DREAM Challenge dataset to a phrase of concept names.
    
```
>> head -5 data_dictionary.csv
concept_id,concept_name,table
22274,Neoplasm of uncertain behavior of larynx,condition_occurrence
22281,Sickle cell-hemoglobin SS disease,condition_occurrence
22288,Hereditary elliptocytosis,condition_occurrence
22340,Esophageal varices without bleeding,condition_occurrence
```

- **`titles.txt`**
    - output from using the csvtool on `metadata.csv` in `process.sh` 
```
>> head -5 titles.txt
title
"Clinical features of culture-proven Mycoplasma pneumoniae infections at King Abdulaziz University Hospital, Jeddah, Saudi Arabia"
Nitric oxide: a pro-inflammatory mediator in lung disease?
Surfactant protein-D and pulmonary host defense
Role of endothelin-1 in lung disease
```


**output/**
- **`ids_from_dict.txt`**
    - output from `get_ids_from_dict.py`. Each line is in the format of concept_id, umls_id1, umls_id2, .... Since there could be multiple candidate UMLS terms in each concept name.
```
>> head -5 ids_from_dict.txt
22274,C0027651,C2981607,C1882062,C1368871,C0026640
22281,C0002895,C2699300,C1260595,C0750151,C3273373,C0086981,C1698490,C1420041,C0039101
22288,C0013902,C1862322,C0039730,C0427480,C0037889
22340,C0014867,C0014849,C0155789,C0019080,C0029163,C0426747,C4743774,C0030922
22666,C1963281,C3898969,C4084766,C0042963,C4084767
```

- **`umls_cui_in_titles.txt`**
    - UMLS IDs extracted from each title. One line per title.

```
(base) slin@Santinas-MacBook-Pro output % head -5 umls_cui_in_titles.txt
C3714514,C0948075,C2242472,C0009450,C0699744
C3810607,C0028128,C0028215,C0600437,C2610645,C0024115,C0034049,C0340177,C0546483,C0555214
C0024115,C0034049,C0340177,C0546483,C0555214
C0009450,C3714514,C1556682,C0948075,C1550587
C0007018,C4520712,C2610644,C0007020,C1268558
```

- **`raw_concept_counts.txt`**

```
(base) slin@Santinas-MacBook-Pro output % head -5 raw_concept_counts.txt
('4308394', 58508)
('4309318', 58508)
('432250', 58489)
('4098617', 57753)
('45757690', 57439)
```


## Preliminary results

If we just take a look at the top 5 concept IDs that have the most count, their corresponding concept names are: 
- 4308394,"Infection and inflammatory reaction due to prosthetic device, implant and graft in urinary system",condition_occurrence
- 4309318,"Infection and inflammatory reaction due to prosthetic device, implant and graft in genital tract",condition_occurrence       
- 432250,Disorder due to infection,condition_occurrence
- 4098617,Hemophagocytic lymphohistiocytosis due to infection,condition_occurrence
- 45757690,Infection due to Shiga toxin producing Escherichia coli,condition_occurrence 

They all have something to do with infection, which most likely mean that we extracted the word "infection" 
as some UMLs candidate IDs from these concept names, and that many papers' titles also contain this term.

This is not great because the word seems like very commonly used terms that are not informative of each of the concept name.

**Proposed solution** 
We can use TF-IDF (term frequency - inverse document frequency) technique to assign a weight to each UMLS ID in each concept name.
TF-IDF is a popular technique to identify important terms in a document based on each term's occurrence in other documents. Words that are in a document 
would deem more important if it occurs only a small number of documents, whereas words such as stop-words that appear in almost all documents would
receive a low score. 
We can then define a threshold at which we'll drop the UMLS IDs with low scores. That way we filter for UMLS IDs that are more specific to each concept.
