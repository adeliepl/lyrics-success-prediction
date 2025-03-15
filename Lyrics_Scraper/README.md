# Lyrics Extractor System 🎵

## **Overview**
This project is a **multi-threaded lyrics extractor** that fetches song lyrics from **Lyrics.ovh API** using parallel processing. It divides the dataset into `N` parts, runs multiple threads to extract lyrics efficiently, and saves the final output to a CSV file.

---

## **Features 🚀**
👉 **Multi-threaded Processing** → Uses multiple threads to fetch lyrics in parallel.  
👉 **CSV Processing** → Reads a CSV file with `Song` and `Performer` columns.  
👉 **Logging System** → Generates a log file for successful and failed extractions.  
👉 **Handles Missing Lyrics** → Saves `NULL` in the `Lyrics` column if lyrics are not found.  
👉 **Scalable** → Allows users to specify the number of threads for faster execution.  
👉 **Final Output** → A CSV file with `Song`, `Performer`, and `Lyrics`.

---

## **Installation & Setup 💻**

### **1️⃣ Install Dependencies**
Make sure you have Python installed, then install required dependencies:
```bash
pip install pandas requests tqdm
```

### **2️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/lyrics-extractor.git
cd lyrics-extractor
```

### **3️⃣ Run the Script**
```bash
python lyrics_extractor.py "/path/to/your/csv_file.csv"
```
- Replace `/path/to/your/csv_file.csv` with the actual path of your CSV file.
- Use `--threads` to specify the number of **parallel threads** (default is `4`).

---

## **Usage Guide 🛠️**

### **Input CSV Format**
Your input **CSV file** must have the following columns:
| Song | Performer |
|------|----------|
| Shape of You | Ed Sheeran |
| Rolling in the Deep | Adele |
| Uptown Funk | Bruno Mars |

---

### **How the System Works ⚙️**
1. **Reads the CSV file** and ensures it has the required columns.
2. **Splits the data** into `N` partitions (based on the number of threads).
3. **Each thread processes its partition**, fetching lyrics from **Lyrics.ovh API**.
4. **Stores lyrics in a new column** (`Lyrics`).
5. **Saves results to a new CSV file** and logs success & failure in `lyrics_extraction.log`.

---
