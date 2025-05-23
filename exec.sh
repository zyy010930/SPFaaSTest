#!/bin/bash

for i in {0..999}
do
	/usr/bin/python3 ./train.py ${i}
done

/usr/bin/python3 ./result.py