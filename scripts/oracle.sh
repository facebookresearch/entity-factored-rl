#!/bin/bash

set -e

python collect_demos.py Fetch1Switch-v1 --chain --num_eps 300
python collect_demos.py Fetch2Switch-v1 --chain --num_eps 300
python collect_demos.py Fetch3Switch-v1 --chain --num_eps 300
python collect_demos.py Fetch4Switch-v1 --chain --num_eps 300
python collect_demos.py Fetch5Switch-v1 --chain --num_eps 300
python collect_demos.py Fetch6Switch-v1 --chain --num_eps 300

python collect_demos.py Fetch1Push-v1 --chain --num_eps 300
python collect_demos.py Fetch2Push-v1 --chain --num_eps 300
python collect_demos.py Fetch3Push-v1 --chain --num_eps 300
python collect_demos.py Fetch4Push-v1 --chain --num_eps 300
python collect_demos.py Fetch5Push-v1 --chain --num_eps 300
python collect_demos.py Fetch6Push-v1 --chain --num_eps 300

python collect_demos.py Fetch1Switch1Push-v1 --num_eps 300 --chain
python collect_demos.py Fetch2Switch2Push-v1 --num_eps 300 --chain
python collect_demos.py Fetch3Switch3Push-v1 --num_eps 300 --chain

python collect_demos.py FetchStack2Stage1-v1 --num_eps 300 --chain
python collect_demos.py FetchStack2Stage3-v1 --num_eps 300 --chain
python collect_demos.py FetchStack2StitchOnlyStack-v1 --num_eps 300 --chain

python collect_demos.py Fetch1PushCollide-v1 --num_eps 300 --chain
python collect_demos.py Fetch2PushCollide-v1 --num_eps 300 --chain
python collect_demos.py Fetch3PushCollide-v1 --num_eps 300 --chain
python collect_demos.py Fetch4PushCollide-v1 --num_eps 300 --chain
python collect_demos.py Fetch5PushCollide-v1 --num_eps 300 --chain
python collect_demos.py Fetch6PushCollide-v1 --num_eps 300 --chain

python collect_demos.py Fetch1Switch1PushCollide-v1 --num_eps 300 --chain
python collect_demos.py Fetch2Switch2PushCollide-v1 --num_eps 300 --chain
python collect_demos.py Fetch3Switch3PushCollide-v1 --num_eps 300 --chain