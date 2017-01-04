#!/bin/bash

echo "Starting to convert with multiple branchlists."
echo ""
echo "Branchlist: evt_jet.txt"
python3 get_branches_evt_jet.py
echo "Done."
echo ""
echo "Branchlist: bdt_evt_jets.txt"
python3 get_branches_bdt_evt_jets.py
echo "Done."
echo ""
echo "Branchlist: bdt_and_weights.txt"
python3 get_branches_bdt_and_weights.py
echo "Done."
echo ""
echo "Branchlist: branches_new.txt"
python3 get_branches_branches_new.py
echo "Done."
echo ""
echo "Job done. No errors occured."
