# debug_parser.py
from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML(typ='safe')
data_dir = Path("data")

for file in data_dir.glob("*.yml"):
    print(f"\n{'='*50}\nTesting: {file}\n{'='*50}")
    try:
        content = yaml.load(file)
        print(f"✓ Parsed as: {type(content).__name__}")
        if isinstance(content, dict):
            print(f"✓ Keys: {list(content.keys())}")
        else:
            print(f"✗ ERROR: Content is a scalar value: {content}")
            print(f"✗ This file will cause the TypeError!")
    except Exception as e:
        print(f"✗ Parse error: {e}")

# Also check for hidden files
import os
hidden_files = [f for f in os.listdir("data/") if f.startswith('.')]
if hidden_files:
    print(f"\n⚠️  Found hidden files: {hidden_files}")