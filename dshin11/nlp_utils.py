import nltk


def extract_entities(text):
    tokens = nltk.tokenize.casual_tokenize(text)
    tags = nltk.pos_tag(tokens)
    grammar = "NP: {<JJ>*(<NN>|<NNP>|<NNPS>|<NNS>)+}"
    parser = nltk.RegexpParser(grammar)
    result = parser.parse_all(tags)

    ret = []
    for item in result:
        if not hasattr(item, 'label'):
            assert len(item) == 2
            ret.append(item[0])
            continue
        assert item.label() == "NP"
        s = ' '.join([l[0] for l in item.leaves()])
        ret.append('<span class="np">{}</span>'.format(s))

    return ret