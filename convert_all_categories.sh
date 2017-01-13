#!/bin/bash

echo "Starting to convert with multiple branchlists."
echo ""
echo "Branchlist: bdt.txt"
python3 get_branches/bdt_1.py &
python3 get_branches/bdt_2.py &
python3 get_branches/bdt_3.py &
python3 get_branches/bdt_4.py 
echo "Done."
# echo ""
# echo "Branchlist: evt_jet.txt"
# python3 get_branches/evt_jet_1.py &
# python3 get_branches/evt_jet_2.py &
# python3 get_branches/evt_jet_3.py &
# python3 get_branches/evt_jet_4.py 
# echo "Done."
# echo ""
# echo "Branchlist: bdt_evt_jets.txt"
# python3 get_branches/bdt_evt_jets_1.py &
# python3 get_branches/bdt_evt_jets_2.py &
# python3 get_branches/bdt_evt_jets_3.py &
# python3 get_branches/bdt_evt_jets_4.py 
# echo "Done."
# echo ""
# echo "Branchlist: bdt_and_weights.txt"
# python3 get_branches/bdt_and_weights_1.py &
# python3 get_branches/bdt_and_weights_2.py &
# python3 get_branches/bdt_and_weights_3.py &
# python3 get_branches/bdt_and_weights_4.py 
# echo "Done."
echo ""
echo "Branchlist: branches_new.txt"
python3 get_branches/branches_new_1.py &
python3 get_branches/branches_new_2.py &
python3 get_branches/branches_new_3.py &
python3 get_branches/branches_new_4.py &
echo "Done."
echo ""
echo "Job done. No errors occured."
