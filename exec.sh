#!/bin/bash

for i in {100..299}
do
	python3 ./train.py ${i}
done

python3 ./result.py