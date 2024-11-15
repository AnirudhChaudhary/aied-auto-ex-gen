"""
Created this main.py because I want to have control over what is being run and how all the outputs are being logged.
"""

import subprocess
import os

# Run the brench command
result = subprocess.run(['python3', 'pipeline.py'], stdout=subprocess.PIPE)
output = result.stdout.decode('utf-8')

# Write the output to a file
count = 1
output_path = f'terminal_output_{count}.txt'
while os.path.exists(output_path):
    count += 1
    output_path = f'terminal_output_{count}.txt'
    
with open(output_path, 'w') as f:
    f.write(output)