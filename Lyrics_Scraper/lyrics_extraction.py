import pandas as pd
import requests
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Setup logging
LOG_FILE = "lyrics_extraction.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# Function to split CSV into batch files
def split_csv(input_file, batch_size, output_folder):
    """Splits the input CSV into smaller batch CSV files."""
    df = pd.read_csv(input_file)
    os.makedirs(output_folder, exist_ok=True)

    batch_files = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_file = os.path.join(output_folder, f"batch_{i//batch_size}.csv")
        batch_df.to_csv(batch_file, index=False)
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
def init_extraction(filepath, batch_size=1000, num_threads=4):
    """Splits CSV, runs multi-threaded lyrics extraction, and merges results."""
    
    try:
        # Step 1: Split CSV into batches
        batch_folder = "batches"
        batch_files = split_csv(filepath, batch_size, batch_folder)

        # Step 2: Process each batch in parallel
        output_folder = "processed_batches"
        os.makedirs(output_folder, exist_ok=True)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
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
    parser.add_argument("--batch_size", type=int, default=1000, help="Size of each batch (default=1000)")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use (default=4)")
    
    args = parser.parse_args()
    init_extraction(args.file_path, args.batch_size, args.threads)
