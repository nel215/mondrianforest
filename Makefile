test:
	PYTHONPATH=. py.test ./tests/*_test.py --capture=sys
lint:
	pep8 --config=./pep8 ./
