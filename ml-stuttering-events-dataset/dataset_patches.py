import csv
import os
from dotenv import load_dotenv

def fix_stuttering_is_cool_links(input_file='SEP-28k_episodes.csv'):
    """
    Reads a CSV file of podcast episodes, corrects the audio URLs for the
    "Stuttering_is_Cool" podcast, and writes the result to a new CSV file.

    Args:
        input_file (str): The path to the original CSV file.
    """
    load_dotenv()
    output_file = os.getenv("SEP28K_EPISODE_FILE_NAME")

    try:
        with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
             open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Read the header row and write it to the new file
            header = next(reader)
            writer.writerow(header)

            # Define constants for column indices for clarity
            SHOW_NAME_IDX = 0
            AUDIO_URL_IDX = 2

            # Counter for the number of links fixed
            links_fixed_count = 0

            # Process each row in the input file
            for row in reader:
                # Check if the row belongs to the target podcast show
                if row[SHOW_NAME_IDX] == 'Stuttering_is_Cool':
                    old_url = row[AUDIO_URL_IDX]
                    
                    # Extract the audio filename (e.g., 'cool203.mp3') from the old URL
                    filename = os.path.basename(old_url)
                    
                    # Construct the new, correct URL
                    new_url = f"https://stutteringiscool.com/sound/{filename}"
                    
                    # Update the URL in the row
                    row[AUDIO_URL_IDX] = new_url
                    links_fixed_count += 1

                # Write the (potentially modified) row to the output file
                writer.writerow(row)
        
        print(f"Processing complete.")
        print(f"Successfully fixed {links_fixed_count} links for 'Stuttering_is_Cool'.")
        print(f"Patched data saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        print("Please ensure the script is in the same directory as the CSV file, or provide the full path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Invoke all patches here
if __name__ == '__main__':
    fix_stuttering_is_cool_links()
