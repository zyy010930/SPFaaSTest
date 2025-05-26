#!/bin/bash

for i in {0..199}
do
	python3 ./train.py ${i}
done

python3 ./result.py