

import json

jsonPath = r"data\jingjiu\tianji-etiquette-chinese-v0.1.json"

# Load the JSON data from a file
with open(jsonPath, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract the "output" values
outputs = []
for item in data:
    for conversation in item['conversation']:
        outputs.append(conversation['output'])

# Write the outputs to a text file
with open('./jingjiu.txt', 'w', encoding='utf-8') as output_file:
    for output in outputs:
        output_file.write(output + '\n')

print("Outputs have been extracted to jingjiu.txt.")
