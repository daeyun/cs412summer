import glob
import textwrap
import unidecode
import enchant
import traceback
import warnings
import difflib
import pickle
from pprint import pprint
import re

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io
import os
from os import path
import project.dshin11.html_parser as html_parser
import project.dshin11.web_search as web_search
import project.dshin11.nlp_utils as nlp_utils

dirname = os.path.dirname(os.path.realpath(__file__))


def pdfparser(data):
    with open(data, 'rb') as fp:
        rsrcmgr = PDFResourceManager()
        # retstr = io.StringIO()
        retstr = io.BytesIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = HTMLConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document.

        for page in PDFPage.get_pages(fp):
            interpreter.process_page(page)
            data = retstr.getvalue()

    return data.decode('utf-8')


def pdf_files_to_html():
    files = sorted(glob.glob('/home/daeyun/Dropbox/ipython-notebooks/cs412/project/pdf/**/*.pdf', recursive=True))
    for file in files:
        html = pdfparser(file)
        outdir = '/tmp/parsed/'
        os.makedirs(outdir, exist_ok=True)

        outfile = os.path.join(outdir, os.path.basename(file)) + '.html'
        with open(outfile, 'w') as f:
            f.write(html)
            print(outfile)


import bs4


def header(content):
    lines = content.split('\n')

    title = lines[0].strip()
    li = 1
    while True:
        if title.strip().endswith(':') or title.strip().endswith('and') or title.strip().endswith('in') or title.strip().endswith('for') or title.strip().endswith('with'):
            title += ' ' + lines[li].strip()
            li += 1
        else:
            break

    if '∗' in title:
        title = title.split('∗')[0]
    title = title.strip()

    header = []
    for i, l in enumerate(lines[li:]):
        if 'abstract' in l.lower():
            break
        header.append(l)
        if i > 30:
            print()
            print(content)
            raise RuntimeError('i: {}'.format(i))

    return title, header[1:]


def main2():
    files = sorted(glob.glob('/home/daeyun/Dropbox/ipython-notebooks/cs412/project/txt/**/*.txt', recursive=True))
    for i, file in enumerate(files):
        with open(file, 'r') as f:
            content = f.read()
        title, head = header(content)
        print(i, file)
        print(title)
        if i == 93:
            print()
            print()
            print(content)
            print()
            print()


def resolve_title_conflict():
    pkl_path = path.join(dirname, '../pkl/acm_search_results.pkl')
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    files = sorted(glob.glob(path.join(dirname, '../html/**/*.html'), recursive=True))
    for file in files:
        name = os.path.basename(file)
        item = metadata[name]
        web_search.assert_valid_search_result_parsing(item)

    item = metadata['kdd16-p795.pdf.html']
    parsed = web_search.select_valid_entry(item['html'], item['query'], return_parsed=True, author_name='Wei Chen')
    item.update(parsed)
    print(item['title'])
    print(item['authors'])
    metadata['kdd16-p795.pdf.html'] = item

    item = metadata['kdd16-p885.pdf.html']
    parsed = web_search.select_valid_entry(item['html'], item['query'], return_parsed=True, author_name='Xinran He')
    item.update(parsed)
    print(item['title'])
    print(item['authors'])
    metadata['kdd16-p885.pdf.html'] = item

    with open(pkl_path + '.new', 'wb') as f:
        pickle.dump(metadata, f)


def extend_parsed_entry():
    pkl_path = path.join(dirname, '../pkl/acm_search_results.pkl')
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    files = sorted(glob.glob(path.join(dirname, '../html/**/*.html'), recursive=True))
    for file in files:
        name = os.path.basename(file)
        item = metadata[name]

        details = bs4.BeautifulSoup(item['entry_html'], 'lxml').select('div.details')[0]
        out = html_parser.parse_acm_search_result_entry(details)
        item.update(out)
        metadata[name] = item

    with open(pkl_path, 'wb') as f:
        pickle.dump(metadata, f)


def check_same_name_authors():
    # Shows profile ids with the same author name.
    pkl_path = path.join(dirname, '../pkl/acm_search_results.pkl')
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    files = sorted(glob.glob(path.join(dirname, '../html/**/*.html'), recursive=True))

    authors = {}
    titles = {}
    for file in files:
        name = os.path.basename(file)
        item = metadata[name]
        for author_name, author_id in item['authors']:
            if author_name in authors:
                if authors[author_name] != author_id:
                    print(author_name, authors[author_name], author_id, item['title'], '|', titles[author_name])
            else:
                authors[author_name] = author_id
                titles[author_name] = item['title']


def save_metadata_without_html():
    pkl_path = path.join(dirname, '../pkl/acm_search_results.pkl')
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    files = sorted(glob.glob(path.join(dirname, '../html/**/*.html'), recursive=True))

    for file in files:
        name = os.path.basename(file)
        item = metadata[name]

        del item['html']
        del item['entry_html']

        metadata[name] = item

    out_path = path.join(dirname, '../pkl/metadata.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(metadata, f)


class WordChecker(object):
    def __init__(self):
        self.dictionaries = [
            enchant.Dict('en_US'),
            enchant.Dict('en'),
        ]

    def is_word(self, word):
        for d in self.dictionaries:
            if d.check(word) or d.check(word.lower()) or d.check(word.upper()) or d.check(word.capitalize()):
                return True


def extract_text():
    pkl_path = path.join(dirname, '../pkl/metadata.pkl')
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
    files = sorted(glob.glob(path.join(dirname, '../html/**/*.html'), recursive=True))

    word_checker = WordChecker()

    for file in files:
        name = os.path.basename(file)
        item = metadata[name]

        soup = html_parser.read_file(file)
        raw_text = soup.text

        m = re.findall('^[\s\S]+?(?:\nABSTRACT|\nAbstract|INTRODUCTION|\n[\s\S]{,5}Introduction|(?=\n[^\n]*?the ))', raw_text)
        assert len(m) == 1
        assert m[0].count('\n') < 55, (m[0].count('\n'), m)
        header = m[0]

        text = raw_text[len(header):]
        text = re.sub(r'Page [0-9]+', '', text)
        # text = re.sub(r'\n[0-9]{1,2}\s{,5}[A-Z \-\_\&]+', '', text)
        # remove email addresses
        text = re.sub(r'[\w\-\.]{,50}@([\w\-]+\.)+[\w\-]{2,4}', '', text)

        all_words = re.findall(r'[A-Za-z\u00C0-\u017F][A-Za-z0-9\u00C0-\u017F\-]+[A-Za-z0-9\u00C0-\u017F](?:(?<!-)\n|\b)', text)
        all_words = {unidecode.unidecode(word.strip().lower()) for word in all_words}

        m = re.findall(r'[A-Za-z0-9\u00C0-\u017F\-]+?\-[ \r\n\t]{0,4}\n[A-Za-z0-9\u00C0-\u017F]+', text)
        corrections = {}
        for match in m:
            match_plain = unidecode.unidecode(match)
            w = re.sub(r'\-[ \r\t\n]\n{0,4}', '', match_plain)
            w2 = re.sub(r'\-[ \r\t\n]\n{0,4}', '-', match_plain)

            if w.lower() in all_words:
                corrections[match] = w
                continue
            if w2.lower() in all_words:
                corrections[match] = w2
                continue
            if word_checker.is_word(w):
                corrections[match] = w
                continue
            if word_checker.is_word(w2):
                corrections[match] = w2
                continue

            all_word_parts = True
            parts = w2.split('-')
            for p in parts:
                if len(p) == 0 or not (word_checker.is_word(p) or p.lower() in all_words):
                    all_word_parts = False
                    break
            if all_word_parts:
                corrections[match] = w2
                continue

            corrections[match] = w2

        for before, after in corrections.items():
            text = text.replace(before, after)

        text = unidecode.unidecode(text)

        text = re.sub(r'\([^)]{0,10}\)', '', text)
        text = re.sub(r'\s+', ' ', text)

        text = ' '.join([part for part in text.split(' ') if len(re.findall(r'[^\w\-\"\'\,]', part)) < 4])

        text = re.sub(r'\[[^]]{0,18}\]', '', text)
        text = re.sub(r'\{[^}]{0,18}\}', '', text)
        text = re.sub(r'\[\?\]', '', text)
        text = re.sub(r'(\w) \.', '\g<1>.', text)
        text = re.sub(r'\.\s{1,5}(com|edu|gov|org|net)\b', '.\g<1>', text)
        # url
        text = re.sub(r'^((http[s]?|ftp):\/)?\/?([^:\/\s]+)((\/\w+)*\/)([\w\-\.]+[^#?\s]+)(.*)?(#[\w\-]+)?$', '', text)

        np = nlp_utils.extract_entities(text)

        out_text = ' '.join(np)

        # for item in np:
        #     print(item)
        #
        # np_set = {item for item in np if len(re.sub(r'[^\w]', '', item)) > 1}
        #
        # out_text = text
        #
        # for item in np_set:
        #     try:
        #         out_text = re.sub(r'\b{}\b'.format(item), '<b>{}</b>'.format(item), out_text)
        #     except:
        #         pass
        #
        outpath = path.join(dirname, '../np_text/')
        os.makedirs(outpath, exist_ok=True)
        outpath = path.join(outpath, name)
        with open(outpath, 'w') as f:
            f.write(out_text)
        print(outpath)


def main():
    extract_text()


if __name__ == '__main__':
    main()
