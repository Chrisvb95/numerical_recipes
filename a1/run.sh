#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

# First task of the assignemnt 
echo "Running the first script - NR_a1_1_main.py"
python3 NR_a1_1_main.py > NR_a1_1_main.txt

# Second task of the assignment
echo "Run the second script - NR_a1_2_main.py"
python3 NR_a1_2_main.py > NR_a1_2_main.txt

echo "Run the third script - NR_a1_3_main.py"
python3 NR_a1_3_main.py > NR_a1_3_main.txt

echo "Generating the pdf"

pdflatex tex_main.tex
#bibtex template.aux



