import json
import numpy as np

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def main():
    data = load_data('full_data_out.json')
    print(f"Loaded {len(data['datasets'][0]['examples'])} examples.")

if __name__ == '__main__':
    main()
