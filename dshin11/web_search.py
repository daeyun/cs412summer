import subprocess
import difflib
import warnings
import requests
import bs4
import urllib
import re
import project.dshin11.html_parser as html_parser


def download_citation(title):
    parts = title.split(' ')

    keywords = []
    for part in parts:
        if re.findall(r'[^a-zA-Z0-9\:\.\,\-\_\?\!]', part.strip()):
            continue
        part = re.sub(r'[^a-zA-Z0-9\:\.\,\-\_\?\!]', '', part)
        keywords.append(part)
    query = ' '.join(keywords)
    print(query)

    out = subprocess.run([
        'python2',
        '/home/daeyun/git/scholar.py/scholar.py',
        '-c',
        '1',
        '-t',
        '--all="{title}"'.format(title=query),
        '--citation=en'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert len(out.stderr) == 0, out.stderr
    ret = out.stdout.decode('utf-8')
    ret = ret.replace('\r\n', '\n')
    return ret


def acm_metadata(title):
    parts = title.split(' ')

    keywords = []
    for part in parts:
        if re.findall(r'[^a-zA-Z0-9\.\,\-\_\:\?\!]', part.strip()):
            print('Skipping word: ', part)
            continue
        part = re.sub(r'[^a-zA-Z0-9\.\-\_]', '', part)
        keywords.append(part)
    query = ' '.join(keywords)
    print('Searching ACM. query:', query)
    encoded_query = urllib.parse.quote_plus(query)
    url = 'http://dl.acm.org/results.cfm?query={}'.format(encoded_query)

    headers = {
        'User-Agent': 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:12.0) Gecko/20100101 Firefox/12.0',
    }
    response = requests.get(url, headers=headers)

    details = select_valid_entry(response.content, query)
    parsed = html_parser.parse_acm_search_result_entry(details)

    parsed['query'] = query
    parsed['html'] = response.content

    assert_valid_search_result_parsing(parsed)

    return parsed


def select_valid_entry(html_content, query, return_parsed=True, author_name=None):
    soup = bs4.BeautifulSoup(html_content, 'lxml')
    details_list = soup.find_all(name='div', class_='details')
    found = None
    for details in details_list:
        try:
            parsed = html_parser.parse_acm_search_result_entry(details)

            if author_name is not None:
                valid = False
                for a, aid in parsed['authors']:
                    if author_name.lower() == a.lower():
                        valid = True
                        break
                if not valid:
                    continue

            parsed['query'] = query
            assert_valid_search_result_parsing(parsed)
        except (AssertionError, IndexError) as ex:
            continue
        else:
            if return_parsed:
                found = parsed
            else:
                found = details
            break
    return found


def assert_valid_search_result_parsing(item):
    assert (item['source'].startswith("KDD '15") or item['source'].startswith("KDD '16")), item['source']
    diff_ratio = difflib.SequenceMatcher(None, item['title'].lower(), item['query'].lower()).ratio()
    assert diff_ratio >= 0.45, (diff_ratio, item['title'], item['query'])
