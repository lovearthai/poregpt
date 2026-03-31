#!/usr/bin/env python3

import sys
import csv
import numpy as np

def process_data(tsv_file, npy_file, output_csv):
    """
    Reads TSV and NPY files, processes the NPY arrays, and writes combined data to a CSV.

    Args:
        tsv_file (str): Path to the input TSV file.
        npy_file (str): Path to the input NPY file.
        output_csv (str): Path to the output CSV file.
    """
    try:
        # Load the NPY array
        print(f"Loading NPY file: {npy_file}")
        reference_arrays = np.load(npy_file)
        print(f"Loaded array shape: {reference_arrays.shape}, dtype: {reference_arrays.dtype}")

        # Check if the number of rows match
        with open(tsv_file, 'r', newline='', encoding='utf-8') as f:
            # Count rows efficiently
            num_rows = sum(1 for line in f) - 1 # Subtract 1 for header

        if reference_arrays.shape[0] != num_rows:
            raise ValueError(f"Row count mismatch: TSV has {num_rows} data rows, "
                             f"NPY has {reference_arrays.shape[0]} rows.")

        print(f"Processing {num_rows} rows...")

        with open(tsv_file, 'r', newline='', encoding='utf-8') as tsv_f, \
             open(output_csv, 'w', newline='', encoding='utf-8') as csv_f:

            # Create TSV reader
            tsv_reader = csv.reader(tsv_f, delimiter='\t')

            # Create CSV writer
            fieldnames = ['fast5', 'read_id', 'chunk_start', 'chunk_size', 'alignment_identity', 'bases']
            csv_writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
            csv_writer.writeheader()

            # Read and skip header from TSV
            header = next(tsv_reader)
            if header[:3] != ['filename', 'read_id', 'chunk_start']:
                 print(f"Warning: Unexpected TSV header format: {header[:3]}. Expected ['filename', 'read_id', 'chunk_start', ...]. Proceeding anyway.", file=sys.stderr)

            # Iterate through TSV rows and corresponding NPY rows
            for i, row in enumerate(tsv_reader):
                if len(row) < 5:
                    print(f"Warning: Skipping malformed TSV row {i+2}: {row}", file=sys.stderr)
                    continue

                # Extract TSV fields
                filename = row[0]
                read_id = row[1]
                chunk_start = row[2]
                chunk_size = row[3]
                alignment_identity = row[4]

                # Get corresponding NPY array
                npy_row = reference_arrays[i]

                # Process the NPY array: remove zeros and convert to string
                # Filter out zeros
                non_zero_values = npy_row[npy_row != 0]
                # Convert to string representation (e.g., [1, 2, 3] -> "123")
                bases_string = ''.join(map(str, non_zero_values))

                # Create dictionary for writing
                output_row = {
                    'fast5': filename,
                    'read_id': read_id,
                    'chunk_start': chunk_start,
                    'chunk_size': chunk_size,
                    'alignment_identity': alignment_identity,
                    'bases': bases_string
                }

                # Write the row to CSV
                csv_writer.writerow(output_row)

                # Optional: Print progress every 10000 rows
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i+1}/{num_rows} rows...")

        print(f"Processing complete. Output saved to {output_csv}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # Define your input and output files here
    tsv_filename = 'out_summary.processed.tsv'
    npy_filename = 'references.npy'
    output_csv_filename = 'output.csv' # Change this to your desired output name

    process_data(tsv_filename, npy_filename, output_csv_filename)
