import pandas as pd
import requests
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Setup logging
LOG_FILE = "lyrics_extraction.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Fixed number of threads
NUM_THREADS = 4

# Function to fetch lyrics
def get_lyrics_lyricsovh(song, artist):
    """Fetch lyrics using Lyrics.ovh API."""
    url = f"https://api.lyrics.ovh/v1/{artist}/{song}"
    response = requests.get(url)

    if response.status_code == 200:
        lyrics = response.json().get("lyrics", "").strip()
        if lyrics:
            logging.info(f"SUCCESS: Lyrics found for '{song}' by '{artist}'")
            return lyrics
    logging.warning(f"FAILED: Lyrics not found for '{song}' by '{artist}'")
    return None  # Returns None instead of "Lyrics not found."

# Function to process a batch CSV
def process_batch(batch_file, output_folder):
    """Reads a batch CSV, extracts lyrics, and saves the result."""
    df = pd.read_csv(batch_file)

    # Ensure Lyrics column exists
    df["Lyrics"] = None  

    for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {batch_file}"):
        song_name = row["Song"]
        artist = row["Performer"]

        lyrics = get_lyrics_lyricsovh(song_name, artist)
        df.at[i, "Lyrics"] = lyrics  # Store lyrics safely

    # Save the processed batch CSV
    output_file = os.path.join(output_folder, os.path.basename(batch_file))
    df.to_csv(output_file, index=False)
    return output_file

# Function to split CSV into exactly 4 equal parts
def split_csv(input_file, output_folder):
    """Splits the input CSV into 4 equal batch CSV files."""
    df = pd.read_csv(input_file)
    os.makedirs(output_folder, exist_ok=True)

    batch_files = []
    partitions = [df.iloc[i::NUM_THREADS] for i in range(NUM_THREADS)]  # Divide into 4 parts

    for i, partition in enumerate(partitions):
        batch_file = os.path.join(output_folder, f"batch_{i}.csv")
        partition.to_csv(batch_file, index=False)
        batch_files.append(batch_file)

    return batch_files

# Function to merge all batch CSVs
def merge_batches(batch_folder, final_output):
    """Merges all processed batch CSVs into a final output CSV."""
    batch_files = [os.path.join(batch_folder, f) for f in os.listdir(batch_folder) if f.endswith(".csv")]
    df_list = [pd.read_csv(f) for f in batch_files]
    
    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_csv(final_output, index=False)
    print(f"Final merged CSV saved at: {final_output}")

# Main function
def init_extraction(filepath):
    """Splits CSV, runs multi-threaded lyrics extraction, and merges results."""
    
    try:
        # Step 1: Split CSV into 4 equal parts
        batch_folder = "batches"
        batch_files = split_csv(filepath, batch_folder)

        # Step 2: Process each batch in parallel
        output_folder = "processed_batches"
        os.makedirs(output_folder, exist_ok=True)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            processed_files = list(executor.map(lambda f: process_batch(f, output_folder), batch_files))

        # Step 3: Merge all processed batch files
        final_output = "final_scraped_lyrics.csv"
        merge_batches(output_folder, final_output)

        print(f"Lyrics extraction completed! Final file: {final_output}")
        print(f"Log file generated at: {LOG_FILE}")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}")

# Run via CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract lyrics from a CSV file using multi-threading and batch processing.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing songs")
    
    args = parser.parse_args()
    init_extraction(args.file_path)
