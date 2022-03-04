#!/bin/bash

set -e

python render_viz.py --env_name Fetch1Push-v1 Fetch3Push-v1 Fetch6Push-v1 --run_id 3khi7siu 3ly9m71w
python render_viz.py --env_name Fetch1Switch-v1 Fetch3Switch-v1 Fetch6Switch-v1 --run_id 2mb98s47 1sfvs6m4
python render_viz.py --env_name Fetch2Switch2Push-v1 Fetch3Switch3Push-v1 --run_id 2uo2uaqh rvwv2fkn
python render_viz.py --env_name FetchStack2Stage1-v1 FetchStack2StitchOnlyStack-v1 FetchStack2Stage3-v1 --run_id 2pw2m40g pflc1mye