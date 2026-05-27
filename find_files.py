import os

def find_files():
    found = []
    # Skip directories that definitely won't contain these files to speed up search
    skip_dirs = {
        'windows', 'program files', 'program files (x86)', 'appdata', 
        '.gemini', 'inetpub', 'intel', 'microsoft', 'package cache'
    }
    
    print("Starting global C drive search...")
    for root, dirs, files in os.walk("C:\\"):
        # Modify dirs in-place to skip unwanted directories
        dirs[:] = [d for d in dirs if d.lower() not in skip_dirs and not d.startswith('.')]
        
        for file in files:
            file_lower = file.lower()
            if "enem" in file_lower:
                path = os.path.join(root, file)
                print(f"Found: {path}")
                found.append(path)
                
    print(f"\nTotal found: {len(found)}")

if __name__ == "__main__":
    find_files()
