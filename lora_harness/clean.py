import json
import re
import argparse
import sys

def clean_patch_content(content):
    """
    Cleans model output by:
    1. Stripping Markdown code fences.
    2. Splitting into distinct file patches.
    3. Deduplicating patches (preventing the 'stuttering' loop).
    4. Removing non-diff 'chatter' lines.
    """
    if not content:
        return ""

    # 1. Remove Markdown code fences entirely
    content = re.sub(r'^```\w*\s*$', '', content, flags=re.MULTILINE)
    
    # 2. Split content into blocks using Lookahead
    # This splits BEFORE 'diff --git', keeping the delimiter as the start of the new block
    parts = re.split(r'(?=diff --git )', content)
    
    unique_patches = []
    seen_files = set()

    for part in parts:
        # Skip empty parts or parts that are just preamble (don't start with diff)
        if not part.strip().startswith("diff --git"):
            continue

        # Extract filename from header for deduplication
        # Format: diff --git a/path/to/file.py b/path/to/file.py
        match = re.search(r'diff --git a/(.*?) b/', part)
        if match:
            filename = match.group(1).strip()
            
            if filename in seen_files:
                continue # Skip duplicate
            
            seen_files.add(filename)
            
            # 3. Clean "Chatter" from within the block
            # Filter out lines that don't look like valid patch lines
            # Valid starts: "diff", "index", "---", "+++", "@@", " ", "+", "-", "\" (No newline)
            valid_lines = []
            lines = part.splitlines()
            
            # Always keep the first 4 lines (Header info) to be safe
            # Then check the rest
            for i, line in enumerate(lines):
                if i < 4:
                    valid_lines.append(line)
                    continue
                
                # Check for valid first char
                if not line: 
                    # Empty lines are allowed
                    valid_lines.append(line)
                    continue
                    
                first_char = line[0]
                if first_char in ['@', ' ', '+', '-', '\\', 'd', 'i']:
                    # 'd' for diff, 'i' for index (in case header is long), others for content
                    valid_lines.append(line)
            
            unique_patches.append("\n".join(valid_lines).rstrip())
    
    return "\n".join(unique_patches).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input .jsonl file")
    parser.add_argument("output_file", help="Path to output cleaned .jsonl file")
    args = parser.parse_args()

    print(f"Cleaning {args.input_file} -> {args.output_file}...")
    
    count = 0
    fixed_count = 0
    
    try:
        with open(args.input_file, 'r') as f_in, open(args.output_file, 'w') as f_out:
            for line in f_in:
                if not line.strip(): continue
                
                try:
                    data = json.loads(line)
                    if 'model_patch' in data:
                        original = data['model_patch']
                        cleaned = clean_patch_content(original)
                        
                        if original != cleaned:
                            fixed_count += 1
                            
                        data['model_patch'] = cleaned
                    
                    f_out.write(json.dumps(data) + '\n')
                    count += 1
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        print(f"Error: File {args.input_file} not found.")
        sys.exit(1)

    print(f"Processed {count} records.")
    print(f"Cleaned/Deduplicated {fixed_count} records.")

if __name__ == "__main__":
    main()