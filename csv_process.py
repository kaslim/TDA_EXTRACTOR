import csv

# Since the CSV reading using pandas did not work, we'll use the csv module
# and manually parse each [a, b] pair in the array
processed_results = []

# Open the CSV file for reading
with open('D:/大四第二学期/MTH301/all_results2.csv', 'r') as file:
    csv_reader = csv.reader(file)

    # Process each row
    for row in csv_reader:
        # Extract the name and label
        name = row[0]
        label = row[1]
        # Process each [a, b] pair
        vector = [b - a for a, b in (eval(pair) for pair in row[2:])]
        # Append the processed row with the name, label, and the new vector
        processed_results.append([name, label] + vector)

# Now we will write the processed data to a new CSV file
output_file_path = 'D:/大四第二学期/MTH301/training-b.csv'
with open(output_file_path, 'w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(processed_results)