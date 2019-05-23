#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

# First task of the assignemnt 
#echo "Running the first script - a2_1.py"
#python3 a2_1.py > a2_1.txt

# Second task of the assignment
#echo "Run the second script - a2_2.py"
#python3 a2_2.py > a2_2.txt

# Third task of the assignment
#echo "Run the second script - a2_3.py"
#python3 a2_3.py > a2_3.txt

# Third task of the assignment
echo "Run the second script - a2_4.py"
python3 a2_4.py > a2_4.txt

echo "Generating the pdf"

#pdflatex tex_main.tex
#bibtex template.aux



