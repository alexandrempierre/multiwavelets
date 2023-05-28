LOC = .venv
PYTHON = $(LOC)/bin/python
PIP = $(LOC)/bin/pip


run: $(LOC)/bin/activate
	echo "Quando implementado, esse comando deve rodar o programa"


$(LOC)/bin/activate: requirements.txt
	python3 -m venv $(LOC)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt


clean:
	rm -rf __pycache__
	rm -rf .venv