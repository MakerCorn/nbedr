import os
import random


def split_jsonl_file(file_path, max_size=199_000_000):
    """Splits a .jsonl file into multiple parts, each not exceeding max_size bytes.

    Args:
        file_path (str): Path to the .jsonl file to split.
        max_size (int): Maximum size in bytes for each part file.

    Returns:
        list: List of created part file paths.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_number = 1
    file_size = 0
    part_file = None
    part_file_name = f"{filename}_part_{file_number}.jsonl"
    created_files = []

    with open(file_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line_size = len(line.encode("utf-8"))
            if file_size + line_size > max_size or part_file is None:
                if part_file:
                    part_file.close()
                part_file_name = f"{filename}_part_{file_number}.jsonl"
                created_files.append(part_file_name)
                part_file = open(part_file_name, "w", encoding="utf-8")
                file_number += 1
                file_size = 0
            part_file.write(line)
            file_size += line_size
    if part_file:
        part_file.close()
    print("Split completed.")
    return created_files


def extract_random_jsonl_rows(file_path, num_rows, output_file):
    """Extracts a given number of random rows from a .jsonl file and saves them to another file.

    Args:
        file_path (str): Path to the .jsonl file to sample from.
        num_rows (int): Number of random rows to extract.
        output_file (str): Path to the output file to save the sampled rows.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    if num_rows > len(lines):
        raise ValueError(f"Requested {num_rows} rows, but file only contains {len(lines)} lines.")
    sampled_lines = random.sample(lines, num_rows)
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(sampled_lines)
    print(f"Extracted {num_rows} random rows to {output_file}.")


def format_file_path(path):
    """Formats a file path string for consistency across the application.

    Args:
        path (str): The file path to format.

    Returns:
        str: The formatted file path.
    """
    # ...existing code...


def read_file_lines(file_path):
    """Reads all lines from a file and returns them as a list.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        list: A list of lines read from the file.
    """
    # ...existing code...


# Example usage in a notebook:
# from scripts.split_files import split_jsonl_file
# split_jsonl_file('yourfile.jsonl')
