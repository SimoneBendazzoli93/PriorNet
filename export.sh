#!/usr/bin/env bash

./build.sh

docker save priornet | gzip -c > PriorNet.tar.gz
