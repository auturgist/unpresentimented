.PHONY: clean

PYTHON_INTERPRETER = python


requirements: 
	pip install -r requirements.txt


vader: 
	$(PYTHON_INTERPRETER) -m nltk.downloader vader_lexicon