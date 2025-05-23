#!/bin/bash

for i in {0..7154}
do
	python ./train.py ${i}
done

python ./result.py