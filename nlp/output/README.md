This folder contains output from the scripts. There are files with concept and counts in decending orders, and the corresponding files with just the concept ids for easier data injestion for model training.


## intermediate files - input to produce final files

**ids_from_dict.txt**
- Contains the UMLS IDs from each concepts.
```
22274,C0027651,C2981607,C1882062,C1368871,C0026640
22281,C0002895,C2699300,C1260595,C0750151,C3273373,C0086981,C1698490,C1420041,C0039101
22288,C0013902,C1862322,C0039730,C0427480,C0037889
```

**umls_cui_in_abstracts.txt**
- Each line contains the UMLS IDS present in one abstract

```
C3714514,C0948075,C2242472,C0009450,C0699744,C3714514,C0948075,C2242472,C0009450,C0699744,C0206171,C0694549,C0678209,C3714514,C0009450,C1556682,C0948075,C1550587,C0032285,C1535939,C0155862,C3714636,C0032302,C0006271,C0006274,C0001311,C0006270,C0006272,C0010200,C3274924,....
```

**umls_cui_in_titles.txt**
- Each line contains the UMLS IDS present in one title
```
C3714514,C0948075,C2242472,C0009450,C0699744
C3810607,C0028128,C0028215,C0600437,C2610645,C0024115,C0034049,C0340177,C0546483,C0555214
...
```


## final concept weights (descending order)

### Raw count
**raw_concept_counts_titles.txt**
- simply the raw aggregated counts of UMLS occurrences for each concept ids
```
('4308394', 58508)
('4309318', 58508)
...
```


### Use only specific UMLS IDs
A UMLS ID can appear in multiple concepts (up to about ~600 concepts). The aggregated counts here are done on only those UMLS ID that only appear in one concept. So we only aggreagted the counts of UMLS IDs that are specific to one concept.

`_IDONLY` is simply the corresponding files but with only the concept IDs.

For those in abstracts 
**concept_counts_filtered_abstracts.txt**

```
('1792515', 16143)
('4048809', 14910)
('4230221', 13077)
('312437', 12829)
...
```

**concept_counts_filtered_abstracts_IDONLY.txt**

```
1792515
4048809
4230221
312437
...
```

For those in titles
- **concept_counts_filtered_titles.txt**
- **concept_counts_filtered_titles_IDONLY.txt**


### TFIDF 
The count of each UML ID count is normalized by taking the log of it and multiply by inverse document frequency.  log(t_d) * log(N/n_t)
where t_d is the term frequency (counts). N is total number of concepts. n_t is the number of concepts that UMLS ID appear in.

`_IDONLY` is simply the corresponding files but with only the concept IDs.

Those in abstracts
- **concept_counts_abstract_tfidf.txt**
- **concept_counts_abstract_tfidf_IDONLY.txt**

In titles
- **concept_counts_title_tfidf.txt**
- **concept_counts_title_tfidf_IDONLY.txt**

