import re
from pprint import pprint
import difflib
import warnings
import bs4


def read_file(name):
    with open(name, 'r') as f:
        content = f.read()

    soup = bs4.BeautifulSoup(content, 'lxml')
    return soup


def fontsize_from_tag(tag):
    style = tag.attrs['style']
    assert (isinstance(style, str)), type(style)
    m = re.findall('font-size.*?([0-9]+)(?:$|[^0-9])', style)
    assert len(m) <= 1, m
    if len(m) == 0:
        return None
    return int(m[0])


def find_title(soup, tags=None):
    if tags is None:
        tags = soup.find_all('span')
    title = []
    title_size = None
    for i, tag in enumerate(tags):
        size = fontsize_from_tag(tag)
        if len(tag.text.strip()) == 0:
            continue
        if len(title) == 0:
            if size is not None and size > 8:
                assert i < 10
                title_size = size
                title.append(tag.text.strip())
        elif (len(''.join(title)) < 7):
            assert i < 10
            title_size = size
            title.append(tag.text.strip())
        elif size == title_size:
            title.append(tag.text.strip())
            assert i < 100
        else:
            assert i < 100
            break

    if len(title) == 0:
        return None
    assert len(title) < 6, title

    # edge cases:
    if len(title) > 1:
        if title[1].strip() == 'TimeMachine:':
            title = [title[1], title[0]] + title[2:]

    ret = ' '.join(title).strip()
    ret = ret.replace('\n', ' ')
    ret = re.sub(r'^([b-zB-Z]) ', r'\g<1>', ret)
    ret = re.sub(r' :', r':', ret)
    ret = re.sub(r' +', r' ', ret)
    ret = re.sub(r':([A-Z])', r': \g<1>', ret)
    return ret


def find_names(soup):
    # file = '/tmp/parsed/kdd16-p955.pdf.html'
    # soup = html_parser.read_file(file)
    tags = soup.find_all('span')
    title = find_title(soup, tags=tags)

    names = []
    for tag in tags:
        if 'abstract' in tag.text.lower():
            break
        name = tag.text.strip()
        if name in title:
            continue
        if ' ' not in name:
            continue

        exclude_pattern = (r'([\@\{\}\(\)\:\;\%\$\#\!]|of|University|[0-9]{2,}|USA|China|ETH|Group|Research'
                           r'|for |Intel|Center|National|Italy|NY|IBM|Dept|Inc\.|Labs|Hong Kong' r')')

        if name.count(' ') > 3 or name.count('\t') > 1:
            for subname in re.split(r'  +', name):
                parts = subname.split(' ')
                if len(parts) <= 1 or len(parts) > 4:
                    continue
                if re.findall(exclude_pattern, subname):
                    continue
                names.append(subname)
        else:
            parts = name.split(' ')
            if len(parts) <= 1 or len(parts) > 4:
                continue
            if re.findall(exclude_pattern, name):
                continue
            names.append(name)

    ret = []
    for name in names:
        name = re.sub(r'(and|,)', '', name)
        ret.append(name.strip())

    return ret


def parse_acm_search_result_entry(details: bs4.Tag):
    assert isinstance(details, bs4.Tag)

    title = details.find(name='div', class_='title').find('a').text

    source = details.select('div.source span:nth-of-type(2)')[0].text

    citation_count = int(re.findall('[0-9]+', details.select('span.citedCount')[0].text)[0])
    downloads_12mo = int(re.findall('[0-9]+', details.select('span.download12Months')[0].text)[0])
    downloads_overall = int(re.findall('[0-9]+', details.select('span.downloadAll')[0].text)[0])
    assert downloads_overall > 0, downloads_overall

    try:
        keywords_div = details.select_one('div.kw')  # type: bs4.Tag
        keywords_str = keywords_div.text.replace('Keywords:', '').strip()
        keywords = [item.strip().lower() for item in keywords_str.split(',')]
    except AttributeError:
        keywords = []

    ref_parent = str(details.find('strong', string='References').find_parent())
    print(title)
    m = re.findall(r'<strong>\s*?References\s*?</strong>[\s\S]+?<br\s*/>\s+([\s\S]+?)(?:<strong>|</div>)', ref_parent)[0]
    ref_lines = re.split(r'<br\s*/>', m)
    ref_lines = [re.sub(r'<\/?[a-zA-Z]{2,6}>', '', line).strip() for line in ref_lines]
    ref_lines = [line for line in ref_lines if len(line) > 0]

    authors = []
    for item in details.select('div.authors a'):
        author_name = item.text
        author_id = re.findall('id=([0-9a-zA-Z]+)\&', item.attrs['href'])[0]
        authors.append((author_name, author_id))

    return {
        'authors': authors,
        'source': source,
        'title': title,
        'citation_count': citation_count,
        'downloads_12mo': downloads_12mo,
        'downloads_overall': downloads_overall,
        'references': ref_lines,
        'keywords': keywords,
        'entry_html': str(details),
    }
