#!/bin/bash

set -e

python collect_demos.py Fetch3Switch-v1 --num_eps 1000
python collect_demos.py Fetch3Switch-v1 --num_eps 2000
python collect_demos.py Fetch3Switch-v1 --num_eps 3000
python collect_demos.py Fetch3Switch-v1 --num_eps 4000
python collect_demos.py Fetch3Switch-v1 --num_eps 5000

python collect_demos.py Fetch3Push-v1 --num_eps 1000
python collect_demos.py Fetch3Push-v1 --num_eps 2000
python collect_demos.py Fetch3Push-v1 --num_eps 3000
python collect_demos.py Fetch3Push-v1 --num_eps 4000
python collect_demos.py Fetch3Push-v1 --num_eps 5000

python collect_demos.py Fetch2Switch2Push-v1 --num_eps 1000
python collect_demos.py Fetch2Switch2Push-v1 --num_eps 2000
python collect_demos.py Fetch2Switch2Push-v1 --num_eps 3000
python collect_demos.py Fetch2Switch2Push-v1 --num_eps 4000
python collect_demos.py Fetch2Switch2Push-v1 --num_eps 5000
