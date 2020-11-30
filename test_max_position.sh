#/bin/bash
title=$(python3 pdftitle.py -p jeong_et_al.pdf -a max_position)
echo "$title"
if [ $? -eq 0 ]; then
  if [ ! "$title" = "Eﬃcient Atomic-Resolution Uncertainty Estimation for Neural Network Potentials Using a Replica Ensemble " ]; then
    exit 1
  fi
  exit 0
else
  exit 1
fi
