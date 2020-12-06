import argparse
import timeit

# return all the ids
def get_concept_ids(concept_text):
    entities = nlp(concept_text)
    ids = [[id[0] for id in concepts._.kb_ents] for concepts in entities.ents if len(concepts._.kb_ents) > 0]
    return list(itertools.chain.from_iterable(ids))


def build_dict(infile: str):
    dictionary = {}
    with open(infile, 'r') as f:
        for line in f:
            tokens = line.split(',')
            concept_id = tokens[0]
            text = ','.join(tokens[1:])
            dictionary[concept_id] = text
    return dictionary

def get_top_concepts(infile: str, n: int):
    ids = []
    
    with open(infile, 'r') as f:
        for line in f:
            id = line.strip()
            ids.append(id)
            if len(ids) == n:
                break;
    return ids

def get_names(dictionary, ids):
    for id in ids:
        print(id, dictionary[id])

if __name__ == "__main__":
    # Arguments for the command line
    parser = argparse.ArgumentParser(description='Inspect the concept names of a list of concpet ids')
    parser.add_argument('--dict', type=str, help="dictionary file", default='../input/data_dictionary.csv')
    parser.add_argument('--ids', type=str, help='file with the list of concept IDs')
    parser.add_argument('--n', type=int, help='number of top concept ids to get the names for.', default=-1)
    args = parser.parse_args()

    t = timeit.default_timer()

    # build a dictionary from dictionary file
    dictionary = build_dict(args.dict)

    # get the top n concepts
    ids = get_top_concepts(args.ids, args.n)
    
    # print them out one by one
    get_names(dictionary, ids)

    print("\nTime took to run", timeit.default_timer() - t)