:: Run tests and generate report

flake8 %~dp0..\tafra
mypy %~dp0..\tafra

pytest
