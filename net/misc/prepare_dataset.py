import argparse
import os

from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString, Comment

import util
import re


def process_bp(doc):
    """Process HTML files annotated by BoilerPipe. We ignore nodes labeled as "comment"."""
    for node in doc.find_all(attrs={'__prediction': True}):
        node['__boilernet_is_content'] = True
        del node['__prediction']


def process_other(source, expected):
    """Process manually annotated HTML files. We ignore nodes labeled as "comment"."""
    # get all the Text elements in the source and expected documents
    source_nodes = get_text_nodes(source.find_all('html')[0])
    expected_nodes = get_text_nodes(expected.find_all('html')[0])

    for source_node in source_nodes:
        source_string = re.sub(
            r'\s+', ' ', source_node.string.strip().replace('\n', ' '))
        # find the corresponding node in the expected
        for expected_node in expected_nodes:
            expected_string = re.sub(
                r'\s+', ' ', expected_node.string.strip().replace('\n', ' '))
            if source_string == expected_string:
                source_node.parent['__boilernet_is_content'] = True
                break


def get_text_nodes(node):
    """Return all Text nodes in a BS4 tree."""
    result = []
    for c in node.children:
        if isinstance(c, NavigableString):
            if c.string is not None and c.string.strip():
                result.append(c)
        elif c.name not in util.TAGS_TO_IGNORE:
            result.extend(get_text_nodes(c))
    return result


def process_gn1(doc):
    """Process HTML files from the GN1 dataset."""
    # the classes x-nc-sel0, x-nc-sel4 and x-nc-sel5 do not seem to indicate content
    nodes = doc.find_all('span', class_='x-nc-sel1') \
        + doc.find_all('span', class_='x-nc-sel2') \
        + doc.find_all('span', class_='x-nc-sel3')
    for node in nodes:
        node['__boilernet_is_content'] = True


def remove_comments(doc):
    """Remove all comments from "doc"."""
    for node in doc.find_all(text=lambda x: isinstance(x, Comment)):
        node.extract()


def process(source, expected, dataset_function):
    """
    Wrap each NavigableString in a <span> tag.
    If the string is content, add a __boilernet_label attribute.
    Remove all HTML comments from the document.
    """
    remove_comments(source)
    dataset_function(source, expected)
    for node, is_content in get_leaves(source.find_all('html')[0]):
        # if the parent node is already a span, we don't add another one
        if node.parent.name == 'span':
            span = node.parent
        else:
            span = source.new_tag('span')
            node.wrap(span)
        if is_content:
            span['__boilernet_label'] = 1
        else:
            span['__boilernet_label'] = 0


def get_leaves(node, is_content=False):
    """Return all leaves (NavigableStrings) in a BS4 tree."""
    if node.has_attr('__boilernet_is_content'):
        is_content = True
        del node['__boilernet_is_content']

    result = []
    for c in node.children:
        if isinstance(c, NavigableString) and not isinstance(c, Comment):
            # might be just whitespace
            if c.string is not None and c.string.strip():
                result.append((c, is_content))
        elif c.name is not None:
            if c.name.lower() in util.TAGS_TO_IGNORE:
                # we remove these tags as they are ignored anyway and can make the file very large
                c.extract()
            else:
                result.extend(get_leaves(c, is_content))
    return result


def main():
    dataset_functions = {'bp': process_bp,
                         'gn1': process_gn1, 'other': process_other}
    ap = argparse.ArgumentParser()
    ap.add_argument('SOURCE', help='source directory (html files)')
    ap.add_argument('EXPECTED', help='expected directory')
    ap.add_argument('OUTPUT', help='Output directory')
    ap.add_argument('DATASET', choices=dataset_functions.keys(),
                    help='Dataset type')
    ap.add_argument('--prefix', help='Add a prefix to the file names.')
    args = ap.parse_args()
    os.makedirs(args.OUTPUT, exist_ok=True)

    i = 0
    source = util.get_filenames(args.SOURCE, '.html')
    expected = util.get_filenames(args.EXPECTED, '.html')
    for f in source:
        try:
            # if not f.endswith('gt_1.html'):
            #     i += 1
            #     continue
            with open(f, 'rb') as hfile:
                doc = BeautifulSoup(
                    hfile, features='html5lib', from_encoding='utf-8')
            with open(expected[i], 'rb') as hfile:
                doc2 = BeautifulSoup(
                    hfile, features='html5lib', from_encoding='utf-8')
            # # for some reason, parsing malformed HTML twice works better
            # doc2 = BeautifulSoup(doc.prettify(), features='html5lib')
            process(doc, doc2, dataset_functions[args.DATASET])
            f_name = os.path.basename(f)
            if args.prefix:
                f_name = args.prefix + f_name
            with open(os.path.join(args.OUTPUT, f_name), 'w', encoding='utf-8') as hfile:
                hfile.write(doc.prettify())

            i += 1
        except:
            tqdm.write('error processing {}'.format(f))


if __name__ == '__main__':
    main()
