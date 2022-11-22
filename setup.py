"""
Tafra: a minimalist dataframe

Copyright (c) 2020 Derrick W. Turk and David S. Fulford

Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
Created on April 25, 2020
"""

import os
import sys
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def find_version() -> str:
    v = {}
    with open('tafra/version.py', 'r') as f:
        exec(f.read(), globals(), v)

    return v['__version__']


def get_long_description() -> str:
    # Fix display issues on PyPI caused by RST markup
    with open('README.rst', 'r') as f:
        readme = f.read()

    replacements = [
        '.. automodule:: tafra',
        ':noindex:',
    ]

    subs = [
        r':func:`([a-zA-Z0-9._]+)`',
        r':meth:`([a-zA-Z0-9._]+)`',
    ]

    def replace(s: str) -> str:
        for r in replacements:
            s = s.replace(r, '')
        return s

    lines = []
    with open('docs/versions.rst', 'r') as f:
        iter_f = iter(f)
        _ = next(f)
        for line in f:
            if any(r in line for r in replacements):
                continue
            lines.append(line)

    version_history = ''.join(lines)
    for sub in subs:
        version_history = re.sub(sub, r'\1', version_history)

    return readme + '\n\n' + version_history


version = find_version()

if sys.argv[-1] == 'build':
    print(f'\nBuilding version {version}...\n')
    os.system('rm -r dist\\')  # clean out dist/
    os.system('python setup.py sdist bdist_wheel')
    sys.exit()


setup(
    name='tafra',
    version=version,
    description='Tafra: innards of a dataframe',
    long_description=get_long_description(),
    long_description_content_type="text/x-rst",
    url='https://github.com/petbox-dev/tafra',
    author='David S. Fulford',
    author_email='petbox.dev@gmail.com',
    install_requires=['numpy>=1.17', 'typing_extensions'],
    zip_safe=False,
    packages=['tafra'],
    package_data={
        'tafra': ['py.typed']
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Typing :: Typed'
    ],
    keywords=[
        'tafra', 'dataframe', 'sql', 'group-by', 'aggregation',
        'performance', 'minimalist'
    ],
)
