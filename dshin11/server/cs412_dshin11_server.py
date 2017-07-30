import os
import json
import glob
import re
from os import path
from flask import Flask
from flask import abort
from flask import jsonify
from flask import redirect
from flask import send_from_directory

dirname = path.dirname(path.realpath(__file__))
app = Flask(__name__)


@app.route('/<path:filename>')
def static_base(filename):
    if '.idea' in filename:
        return abort(404)
    return send_from_directory(path.join(app.root_path, '..', '..', 'js'), filename)


@app.route('/static/np_text/<path:filename>')
def static_np_text(filename):
    if not filename.endswith('.html'):
        return abort(404)
    return send_from_directory(path.join(app.root_path, '..', '..', 'np_text'), filename)


@app.route('/np_text')
def static_np_text_filenames():
    filenames = glob.glob(path.join(app.root_path, '..', '..', 'np_text', '*.html'))
    filenames = [path.basename(item).replace('.pdf.html', '') for item in filenames]
    filenames = sorted(filenames)
    return jsonify(filenames)


@app.route('/np_text/<path:name>')
def np_text_data(name):
    filename = path.join(app.root_path, '..', '..', 'np_text', '{}.pdf.html'.format(name))
    with open(filename, 'r') as f:
        content = f.read()

    pattern = re.compile(r'<span class="np">(.*?)</span>')
    iter = pattern.finditer(content)

    content_index = 0
    new_content_index = 0

    spans = []
    for match in iter:
        span = match.span(0)
        inner_span = match.span(1)
        chars_skipped = span[0] - content_index
        new_content_index += chars_skipped

        new_content_span = (new_content_index, new_content_index + inner_span[1] - inner_span[0])
        spans.append(new_content_span)

        content_index = span[1]
        new_content_index = new_content_span[1]

    new_content = pattern.sub('\g<1>', content)

    return jsonify(
        markers=spans,
        content=new_content,
    )


@app.route('/')
def main():
    return send_from_directory(path.join(app.root_path, '..', '..', 'js'), 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6500, debug=True)
