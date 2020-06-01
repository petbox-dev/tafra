#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


echo flake8 ../tafra
flake8 $DIR/../tafra
echo

echo mypy ../tafra
mypy $DIR/../tafra
echo

echo pytest --cov=tafra --cov-report=term-missing --hypothesis-show-statistics -v .
pytest --cov=tafra --cov-report=term-missing --hypothesis-show-statistics -v .
