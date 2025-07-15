import re
import csv
import os


input_path = r"E:\OneDrive - Delft University of Technology\TUD Master\graduation project\ML\UR5\URscript_v2.txt"

if not os.path.isfile(input_path):
    print("❌ The file path is invalid, please check if the file exists.")
    exit()

with open(input_path, "r") as file:
    script_content = file.read()

# extract 6 numbers in movej([...]) 
pattern = r"movej\(\[([^\]]+)\]"
matches = re.findall(pattern, script_content)

formatted_data = []
for match in matches:
    numbers = match.split(",")
    if len(numbers) == 6:
        stripped_numbers = [num.strip() for num in numbers]
        formatted_data.append(stripped_numbers)

output_csv = os.path.join(os.path.dirname(input_path), "movej_positions.csv")
with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(formatted_data)

print(f"✅ Extraction complete, CSV file saved to:{output_csv}")
