#!/bin/bash

while true; do
  # Use grep to check if the output of nvidia-smi contains "python"
  if nvidia-smi | grep -q python; then
    echo "Found 'python' in the output of nvidia-smi. Waiting for 600 seconds..."
    sleep 600
  else
    echo "'python' not found in the output of nvidia-smi. Executing the given command..."
    python train.py
    # Break out of the loop if you want the script to exit after executing the command
    # or remove the break statement to continue checking indefinitely
    break
  fi
done
