import os

m = {'O': 0, 'B-ORG': 1, 'I-ORG': 2, 'B-MISC': 3, 'I-MISC': 4, 'B-PER': 5, 'I-PER': 6, 'B-LOC': 7, 'I-LOC': 8}


def format_file(input_file, output_file):
    result = []
    count = 0
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                result.append("")
                continue
            line = line.split(" ")
            result.append("{}	{}".format(line[0], line[3]))
            count += 1
    print(count)

    with open(output_file, 'w') as f:
        for r in result:
            f.write(r + "\n")


format_file("data/CoNLL-2003/eng.train", "data/CoNLL-2003/train.tsv")
format_file("data/CoNLL-2003/eng.testa", "data/CoNLL-2003/devel.tsv")
format_file("data/CoNLL-2003/eng.testb", "data/CoNLL-2003/test.tsv")
