"""
Tafra: the innards of a dataframe

MIT License

Copyright (c) 2020 Petroleum Engineering Toolbox

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Author
------
Derrick W. Turk
David S. Fulford

Notes
-----
Created on April 25, 2020
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='tafra',
    version='0.1.0',
    description='Tafra: innards of a dataframe',
    long_description = open('README.rst').read(),
    url = 'https://github.com/petbox-dev/tafra',
    author='Derrick W. Turk',
    install_requires=['numpy', 'typing_extensions'],
    zip_safe=False,
    package_data={
        'tafra': ['py.typed']
    },
    packages=['tafra']
)
