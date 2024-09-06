#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 --job <job_number> --extension <extension> [--extension <extension> ...]"
    echo ""
    echo "This script deletes all files with the specified extensions from the current directory and subdirectories"
    echo "if the numeric part of the filename is less than the provided job number."
    echo ""
    echo "Options:"
    echo "  --job <job_number>      The job number for comparison."
    echo "  --extension <extension> Specify file extension(s) to target, e.g., log, csv, etc."
    echo "  --help                  Display this help message and exit."
    echo ""
    echo "Example:"
    echo "  $0 --job 19197989 --extension log --extension csv"
    echo ""
    exit 0
}

# Initialize variables
job_number=""
extensions=()

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --job) job_number="$2"; shift ;;
        --extension) extensions+=("$2"); shift ;;
        --help) show_help ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure job number and at least one extension are provided
if [ -z "$job_number" ]; then
    echo "Error: --job argument is required."
    show_help
    exit 1
fi

if [ ${#extensions[@]} -eq 0 ]; then
    echo "Error: At least one --extension argument is required."
    show_help
    exit 1
fi

# Iterate over each specified extension
for ext in "${extensions[@]}"; do
    echo "Processing .$ext files..."

    # Find and delete files matching the extension and condition, excluding hidden directories
    find . -type d -name '.*' -prune -o -type f -name "*.$ext" ! -path "./.*/*" -print | while read -r file; do
        numeric_part=$(basename "$file" .$ext | grep -o '[0-9]*$')

        # Check if numeric part is not empty before comparing
        if [ -n "$numeric_part" ]; then
            if [ "$numeric_part" -lt "$job_number" ]; then
                rm "$file"
                echo "Deleted $file"
            fi
        else
            echo "No numeric part found for $file, skipping."
        fi
    done
done

echo "Cleanup completed."
