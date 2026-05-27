import os

def find_files():
    search_dirs = [
        "C:\\Users\\Latitude 5410\\OneDrive\\Desktop",
        "C:\\Users\\Latitude 5410\\Downloads",
        "C:\\Users\\Latitude 5410\\Documents",
        "C:\\Users\\Latitude 5410\\OneDrive"
    ]
    
    found = []
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                file_lower = file.lower()
                # Check for csv, xlsx, zip, rar, 7z
                exts = ('.csv', '.xlsx', '.zip', '.rar', '.7z')
                if file_lower.endswith(exts) or 'enem' in file_lower or 'micro' in file_lower:
                    path = os.path.join(root, file)
                    found.append((file, path, os.path.getsize(path)))
                    
    # Sort by size to find large files easily (microdados is usually very large, > 1GB)
    found.sort(key=lambda x: x[2], reverse=True)
    
    print("Files found (top 50 by size):")
    for name, path, size in found[:50]:
        print(f"Name: {name} | Size: {size / (1024*1024):.2f} MB | Path: {path}")

if __name__ == "__main__":
    find_files()
