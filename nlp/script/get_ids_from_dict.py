import argparse
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker
import math
import itertools
import timeit

disease_chemical_model = 'en_ner_bc5cdr_md'

# example code https://github.com/allenai/scispacy
nlp = spacy.load(disease_chemical_model)
abbreviation_pipe = AbbreviationDetector(nlp)
linker = EntityLinker(name="umls")
nlp.add_pipe(linker)


# nuance example:
# example input: 'Sickle cell-hemoglobin SS disease'
# example output: (Sickle, SS)
# each of the two would get 5 umls concept ids in order of relevance score (descending)
# where the most relevant concept id for Sickle is correct ('C0002895', 0.7860167026519775)
# but for SS it's not ('C0039101', 1.0)

# return all the ids
def get_concept_ids(concept_text):
    entities = nlp(concept_text)
    ids = [[id[0] for id in concepts._.kb_ents] for concepts in entities.ents if len(concepts._.kb_ents) > 0]
    return list(itertools.chain.from_iterable(ids))


def record_umls_ids(infile: str, outfile: str, start: int = None, end: int = None):
    index = 0
    if start == None:
        start = 0
    if end == None:
        end = math.inf
    if start > end:
        raise Exception("end index needs to be greater than start index")
    with open(infile, 'r') as f, open(outfile, 'w') as outf:
        t = timeit.default_timer()
        for line in f:
            if index < start:
                index += 1
                continue
            if index > end:
                break

            # handle when concept text has comma in it
            tokens = line.split(",")
            concept_id, concept_text, occurrence_type = tokens[0], ' '.join(tokens[1:-1]), tokens[-1]

            ids = get_concept_ids(concept_text)
            index += 1
            if len(ids) == 0:
                continue
            outf.write("{},{}".format(concept_id, ','.join(ids)))
            outf.write('\n')
            if (index - start) % 100 == 0:
                print("Done with ", index)
                print("\nTime took to run", timeit.default_timer() - t)
                t = timeit.default_timer()


if __name__ == "__main__":
    # Arguments for the command line
    parser = argparse.ArgumentParser(description='Get concept ids from dictionary')
    parser.add_argument('--dict', type=str, help="dictionary file")
    parser.add_argument('--begin', type=int, help='line number to start with')
    parser.add_argument('--end', type=int, help='line number to end at')
    parser.add_argument('--out', type=str, help='name of the outfile, abs or relative path')
    args = parser.parse_args()

    t = timeit.default_timer()
    record_umls_ids(args.dict, args.out, args.begin, args.end)
    print("\nTime took to run", timeit.default_timer() - t)