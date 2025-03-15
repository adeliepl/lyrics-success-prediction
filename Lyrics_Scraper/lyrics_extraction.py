import pandas as pd
import requests
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Set up logging
LOG_FILE = "lyrics_extraction.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

def process_partition(data_partition):
    """Process a partition of the data, fetching lyrics for each song."""
    for i, row in tqdm(data_partition.iterrows(), total=data_partition.shape[0], desc="Fetching Lyrics"):
        song_name = row["Song"]
        artist = row["Performer"]
        
        lyrics = get_lyrics_lyricsovh(song_name, artist)
        data_partition.at[i, "Lyrics"] = lyrics  # Store lyrics in the dataframe
    
    return data_partition

def init_extraction(filepath, num_threads=4):
    """Extracts lyrics from the dataset using multi-threading."""
    
    try:
        # Load the dataset
        df = pd.read_csv(filepath)

        # Ensure only the required columns exist
        df = df[["Song", "Performer"]].copy()
        df["Lyrics"] = None  # Set default value for lyrics as NULL

        # Split data into `num_threads` partitions
        partitions = [df.iloc[i::num_threads] for i in range(num_threads)]

        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(process_partition, partitions)

        # Combine results
        df_final = pd.concat(results)

        # Save results to a new CSV file
        output_path = os.path.join(os.path.dirname(filepath), "scraped_lyrics.csv")
        df_final.to_csv(output_path, index=False)
        
        print(f"Lyrics extraction completed! Saved to {output_path}")
        print(f"Log file generated at: {LOG_FILE}")

    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        logging.error(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}")

# Example Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract lyrics from a CSV file using multi-threading.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing songs")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads to use (default=4)")
    
    args = parser.parse_args()
    init_extraction(args.file_path, args.threads)
