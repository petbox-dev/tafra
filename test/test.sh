#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


echo flake8 ../tafra
flake8 $DIR/../tafra
echo

echo mypy ../tafra
mypy $DIR/../tafra
echo

echo sphinx-build -W -b html -a ../docs ../docs/_build/html
sphinx-build -W -b html -a $DIR/..docs $DIR/..docs/_build/html
echo

echo pytest
pytest
