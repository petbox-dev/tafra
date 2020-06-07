:: Run tests and generate report

flake8 %~dp0..\tafra
mypy %~dp0..\tafra

sphinx-build -W -b html -a %~dp0..\docs %~dp0..\docs\_build\html

pytest
