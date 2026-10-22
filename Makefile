ROOT ?= data/raw/provided_dataset
DS ?= ds-v1
MODEL ?= m-v1
SUITE ?= configs/eval/test_suite_v1.yaml

.PHONY: setup test ingest qc dataset approve train-act train-smolvla eval dashboard status

setup:
	pip install -e ".[dev,dashboard,video]"

test:
	pytest

ingest:
	hackpipe ingest --root $(ROOT)

qc:
	hackpipe qc --root $(ROOT)

dataset:
	hackpipe dataset create

approve:
	hackpipe dataset approve $(DS)

train-act:
	hackpipe train --config configs/train/act.yaml --dataset $(DS)

train-smolvla:
	hackpipe train --config configs/train/smolvla.yaml --dataset $(DS)

eval:
	hackpipe eval --model $(MODEL) --suite $(SUITE)

dashboard:
	streamlit run dashboard/app.py

status:
	hackpipe status
