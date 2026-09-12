import os

# File paths
list_file = "Fonts/list_fonts_read_2016.txt"
search_file = "Fonts/fonts.txt"  

# Step 1: Extract filenames from the list file
with open(list_file, "r") as f:
    filenames = []
    for line in f:
        line = line.strip()
        if line.startswith('"'):
            path = line.split('"')[1]
            filename = os.path.basename(path)
            filenames.append(filename)

# Step 2: Search for filenames in the other file and print matching lines
with open(search_file, "r") as f:
    lines = f.readlines()

for filename in filenames:
    found = False
    for line in lines:
        if filename in line:
            #print(line.strip())
            found = True
    if not found:
        print(f"File not found: {filename}")


list_file = "Fonts/list_fonts_rimes.txt"
search_file = "Fonts/fonts.txt"  

# Step 1: Extract filenames from the list file
with open(list_file, "r") as f:
    filenames = []
    for line in f:
        line = line.strip()
        if line.startswith('"'):
            path = line.split('"')[1]
            filename = os.path.basename(path)
            filenames.append(filename)

# Step 2: Search for filenames in the other file and print matching lines
with open(search_file, "r") as f:
    lines = f.readlines()

for filename in filenames:
    found = False
    for line in lines:
        if filename in line:
            found = True
            #print(line.strip())
    if not found:
        print(f"File not found: {filename}")

list_file = "Fonts/list_fonts_iam.txt"
search_file = "Fonts/fonts.txt"  

# Step 1: Extract filenames from the list file
with open(list_file, "r") as f:
    filenames = []
    for line in f:
        line = line.strip()
        if line.startswith('"'):
            path = line.split('"')[1]
            filename = os.path.basename(path)
            filenames.append(filename)

# Step 2: Search for filenames in the other file and print matching lines
with open(search_file, "r") as f:
    lines = f.readlines()

for filename in filenames:
    found = False
    for line in lines:
        if filename in line:
            found = True
            #print(line.strip())
    if not found:
        print(f"File not found: {filename}")
