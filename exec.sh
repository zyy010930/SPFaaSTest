#!/bin/bash

for i in {0..999}
do
	python ./train.py ${i}
done

python ./result.py