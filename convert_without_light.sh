#!/bin/bash

echo "Starting to convert with multiple branchlists."
echo ""
echo "Branchlist: bdt.txt"
# python3 get_branches/bdt_1_without_light.py &
python3 get_branches/bdt_2_without_light.py &
# python3 get_branches/bdt_3_without_light.py &
# python3 get_branches/bdt_4_without_light.py 
echo "Done."
echo ""
echo "Branchlist: bdt_evt_jets.txt"
# python3 get_branches/bdt_evt_jets_1_without_light.py &
python3 get_branches/bdt_evt_jets_2_without_light.py &
# python3 get_branches/bdt_evt_jets_3_without_light.py &
# python3 get_branches/bdt_evt_jets_4_without_light.py
echo ""
echo "Branchlist: branches_new.txt"
# python3 get_branches/branches_new_1_without_light.py &
python3 get_branches/branches_new_2_without_light.py
# python3 get_branches/branches_new_3_without_light.py &
# python3 get_branches/branches_new_4_without_light.py
echo "Done."
echo ""
echo "Job done. No errors occured."
