#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
  mkdir plots/2Dmovie
  mkdir plots/3Dmovie
  mkdir plots/3Dmovie/xy
  mkdir plots/3Dmovie/xz
  mkdir plots/3Dmovie/yz
fi

# First task of the assignemnt 
#echo "Running: a2_1.py"
#python3 a2_1.py > a2_1.txt

# Second task of the assignment
#echo "Running: a2_2.py"
#python3 a2_2.py > a2_2.txt

# Third task of the assignment
#echo "Running: a2_3.py"
#python3 a2_3.py > a2_3.txt

# Fourth task of the assignment
#echo "Running: a2_4.py"
#python3 a2_4.py > a2_4.txt

# code that makes a movie of the movie frames
#ffmpeg -framerate 30 -pattern_type glob -i "plots/2Dmovie/snap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 2D.mp4

#ffmpeg -framerate 30 -pattern_type glob -i "plots/3Dmovie/xy/snap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 3D_xy.mp4

#ffmpeg -framerate 30 -pattern_type glob -i "plots/3Dmovie/xz/snap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 3D_xz.mp4

#ffmpeg -framerate 30 -pattern_type glob -i "plots/3Dmovie/yz/snap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 3D_yz.mp4

# Fifth task of the assignment
echo "Running: a2_5.py"
python3 a2_5.py > a2_5.txt

#echo "Generating the pdf"

#pdflatex tex_main.tex
#bibtex template.aux



