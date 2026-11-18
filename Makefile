.PHONY: install run ingest test eval
install:
	python -m pip install -r requirements.txt
run:
	python -m grounded serve
ingest:
	python -m grounded ingest samples
test:
	python -m unittest discover -s tests -v
eval:
	python -m grounded.evaluate --output reports/evaluation.json
