import pandas as pd

# Load the CSV
csv_path = "/home/rayen/coqui/vctk_metadata_upd_cleaned.csv"
df = pd.read_csv(csv_path)

# Function to convert Windows path to WSL path
def update_path(windows_path):
    # Define the old Windows base path and new WSL base path
    windows_base = r"C:\Users\rayen\Downloads\VCTK\VCTK-Corpus\VCTK-Corpus"
    wsl_base = "~/coqui/VTCK/"
    
    # Replace the Windows base path with the WSL base path
    # Convert backslashes to forward slashes for WSL
    wsl_path = windows_path.replace(windows_base, wsl_base).replace("\\", "/")
    return wsl_path

# Update the file_path column
df['file_path'] = df['file_path'].apply(update_path)

# Save the updated CSV
output_csv = "/home/rayen/coqui/vctk_metadata_upd_cleaned_wsl.csv"
df.to_csv(output_csv, index=False)
print(f"Updated CSV saved as {output_csv}")

# Verify the first few rows
print("First 5 rows of updated CSV:")
print(df.head())