import os

ttf_dir = "./Fonts/ttf"
list_file = "./Fonts/list_fonts_read_2016.txt"
output_file = "./Fonts/list_fonts_read_2016__updated.txt"  # Change if you want to overwrite

# Get all ttf filenames in the directory
ttf_filenames = set(os.listdir(ttf_dir))

# Read the list file
with open(list_file, "r") as f:
    lines = f.readlines()

updated_lines = []
for line in lines:
    if '"' in line:
        # Extract the path between quotes
        path = line.split('"')[1]
        filename = os.path.basename(path)
        if filename in ttf_filenames:
            # Replace the path
            new_path = f'./Fonts/ttf/{filename}'
            # Replace only the path part
            new_line = line.replace(path, new_path)
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)
    else:
        updated_lines.append(line)

# Write the updated lines to a new file
with open(output_file, "w") as f:
    f.writelines(updated_lines)

print(f"Updated file written to {output_file}")
