"""
Tafra: the innards of a dataframe


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

from tafra import __version__

try:
    from setuptools import setup  # type: ignore
except ImportError:
    from distutils.core import setup


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

if sys.argv[-1] == 'build':
    os.system('python setup.py sdist bdist_wheel')
    sys.exit()


setup(
    name='tafra',
    version=__version__,
    description='Tafra: innards of a dataframe',
    long_description=get_long_description(),
    long_description_content_type="text/x-rst",
    url='https://github.com/petbox-dev/tafra',
    author='David S. Fulford',
    author_email='petbox.dev@gmail.com',
    install_requires=['numpy>=1.17'],
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
        'tafra', 'dataframe', 'sql', 'group-by',
        'performance'
    ],
)
