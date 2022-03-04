#!/bin/bash

set -e

python demo.py --env_name Fetch1Push-v1 Fetch2Push-v1 Fetch3Push-v1 Fetch4Push-v1 Fetch5Push-v1 Fetch6Push-v1 --run_path ayzhong/fetch-her --run_tag eval
python demo.py --env_name Fetch1Switch-v1 Fetch2Switch-v1 Fetch3Switch-v1 Fetch4Switch-v1 Fetch5Switch-v1 Fetch6Switch-v1 --run_path ayzhong/fetch-her --run_tag eval-3s
python demo.py --env_name Fetch1Switch1Push-v1 Fetch2Switch2Push-v1 Fetch3Switch3Push-v1 --run_path ayzhong/fetch-her --run_tag eval-2s2p
python demo.py --env_name Fetch2SwitchOr2PushOnlyCube-v1 Fetch2SwitchOr2PushOnlySwitch-v1 Fetch2Switch2Push-v1 --run_path ayzhong/fetch-her --run_tag eval-2sor2p
python demo.py --env_name FetchStack2Stage1-v1 FetchStack2StitchOnlyStack-v1 FetchStack2Stage3-v1 --run_path ayzhong/fetch-her --run_tag eval-2stack

python demo.py --env_name Fetch1PushCollide-v1 Fetch2PushCollide-v1 Fetch3PushCollide-v1 Fetch4PushCollide-v1 Fetch5PushCollide-v1 Fetch6PushCollide-v1 --run_path ayzhong/fetch-her --run_tag eval-3pcolliide
python demo.py --env_name Fetch1Switch1PushCollide-v1 Fetch2Switch2PushCollide-v1 Fetch3Switch3PushCollide-v1 --run_path ayzhong/fetch-her --run_tag eval-2s2pcollide
python demo.py --env_name Fetch2SwitchOr2PushOnlyCubeCollide-v1 Fetch2SwitchOr2PushOnlySwitchCollide-v1 Fetch2Switch2PushCollide-v1 --run_path ayzhong/fetch-her --run_tag eval-2sor2pcollide