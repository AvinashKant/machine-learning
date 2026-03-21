import csv
import random
import string

def generate_csv(file_name, schema, total_records, null_percent):
    total_cols = len(schema)
    total_cells = total_records * total_cols
    # Calculate how many total cells should be null
    null_count = int(total_cells * (null_percent / 100))
    
    headers = list(schema.keys())
    data = []

    # 1. Generate the initial full dataset based on type
    for _ in range(total_records):
        row = []
        for col_name, col_type in schema.items():
            if col_type == "string":
                val = ''.join(random.choices(string.ascii_letters, k=7))
            elif col_type == "integer":
                val = random.randint(1, 100)
            elif col_type == "boolean":
                val = random.choice([True, False])
            elif col_type == "float":
                val = round(random.uniform(1.0, 100.0), 2)
            else:
                val = "N/A"
            row.append(val)
        data.append(row)

    # 2. Inject random NULL values (Empty Strings)
    cells_modified = 0
    while cells_modified < null_count:
        r = random.randint(0, total_records - 1)
        c = random.randint(0, total_cols - 1)
        
        # Only increment if we haven't already nulled this specific cell
        if data[r][c] != "":
            data[r][c] = "" 
            cells_modified += 1

    # 3. Write to CSV
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

    print(f"File '{file_name}' created: {total_records} rows, {null_percent}% nulls ({null_count} cells).")

input_schema = {
    "name": "string",
    "age": "integer",
    "is_alive": "boolean", # New Boolean Type
    "score": "float"       # Added for extra utility
}

generate_csv(
    file_name="output.csv", 
    schema=input_schema, 
    total_records=10000, 
    null_percent=5
)
