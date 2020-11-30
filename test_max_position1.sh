#/bin/bash
title=$(python3 pdftitle.py -p huynh_et_al.pdf -a max_position)
echo "$title"
if [ $? -eq 0 ]; then
  if [ ! "$title" = "Persistent Covalency and Planarity in the BnAl6 n a n dLiBAl (n = 0 6) Cluster Ions " ]; then
    exit 1
  fi
  exit 0
else
  exit 1
fi
