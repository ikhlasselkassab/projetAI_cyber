import json
import logging

# Configure logging
logging.basicConfig(filename='technique_to_cve_paths.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_json_files():
    """Load all JSON and JSONL data files and return them as a dictionary."""
    json_files = {
        'cve_map_cpe_cwe_score': 'cve_map_cpe_cwe_score.json',
        'cwe_names': 'cwe_names.json',
        'cwe_from_xml': 'cwe_from_xml.json',
        'category_map_cwe': 'category_map_cwe.json',
        'cwe_descriptions': 'cwe_descriptions.json',
        'capec_cwe_mapping': 'capec_cwe_mapping.json',
        'capec_names': 'capec_names.json',
        'capec_from_xml': 'capec_from_xml.json',
        'capec_descriptions': 'capec_descriptions.json',
        'capec_techniques_map': 'capec_technique_map.json',
        'technique_name_map': 'technique_name_map.json',
        'technique_descriptions': 'technique_descriptions.json',
        'technique_detection': 'technique_detection.jsonl',
        'technique_tactic_map': 'technique_tactic_map.json',
        'tactic_id_name_map': 'tactic_id_name_map.json',
        'tactic_descriptions': 'tactic_descriptions.json',
    }
    
    jsonl_files = {
        'technique_mitigation_technique_mapping': 'technique_mitigation_technique_mapping.jsonl',
        'technique_mitigations': 'technique_mitigations.jsonl',
        'technique_detection': 'technique_detection.jsonl',
    }
    
    data = {}
    
    # Load JSON files
    for key, file_name in json_files.items():
        try:
            with open(f'../data-result/{file_name}') as f:
                data[key] = json.load(f)
                logging.info(f'{file_name} loaded successfully.')
        except Exception as e:
            logging.error(f'Error loading {file_name}: {e}')
    
    # Load JSONL files
    for key, file_name in jsonl_files.items():
        data[key] = []
        try:
            with open(f'../data-result/{file_name}') as f:
                for line in f:
                    entry = json.loads(line)
                    data[key].append(entry)
            logging.info(f'{file_name} loaded successfully.')
        except Exception as e:
            logging.error(f'Error loading {file_name}: {e}')
    
    return data

# Load all JSON data
data = load_json_files()

# Assign data to respective variables
cve_data = data['cve_map_cpe_cwe_score']
cwe_names = data['cwe_names']
cwe_from_xml = data['cwe_from_xml']
cwe_categories = data['category_map_cwe']
cwe_description = data['cwe_descriptions']
capec_cwe_mapping = data['capec_cwe_mapping']
capec_names = data['capec_names']
capec_from_xml = data['capec_from_xml']
capec_descriptions = data['capec_descriptions']
capec_techniques_map = data['capec_technique_map']
technique_name_map = data['technique_name_map']
technique_descriptions = data['technique_descriptions']
technique_detection = data['technique_detection']
technique_tactic_map = data['technique_tactic_map']
tactic_id_name_map = data['tactic_id_name_map']
tactic_descriptions = data['tactic_descriptions']
technique_mitigation_technique_mapping = data['technique_mitigation_technique_mapping']
technique_mitigations = data['technique_mitigations']

# Function to fetch technique details
def get_technique_details(technique_id):
    if technique_id in technique_name_map:
        logging.info(f'Fetching details for technique: {technique_id}')
        technique_info = {
            'technique_id': technique_id,
            'technique_name': technique_name_map[technique_id],
            'description': technique_descriptions.get(technique_id, 'No description available.'),
            'mitigations': get_mitigation_details(technique_id)
        }
        return technique_info
    return None

# Function to fetch CAPEC details
def get_capec_details(capec_id):
    if capec_id in capec_names:
        logging.info(f'Fetching details for CAPEC: {capec_id}')
        capec_info = {
            'capec_id': capec_id,
            'capec_name': capec_names[capec_id],
            'description': capec_descriptions.get(capec_id, 'No description available.')
        }
        return capec_info
    return None

# Function to fetch CWEs related to a CAPEC
def get_cwes_for_capec(capec_id):
    cwes = []
    if capec_id in capec_cwe_mapping['capec_cwe']:
        for cwe_id in capec_cwe_mapping['capec_cwe'][capec_id].get('cwes', []):
            cwe_info = {
                'cwe_id': cwe_id,
                'cwe_name': cwe_names.get(cwe_id, 'Unknown CWE'),
                'description': cwe_description.get(cwe_id, {}).get('description', 'No description available.'),
                'categories': get_categories(cwe_id)
            }
            cwes.append(cwe_info)
    return cwes

# Function to get all categories a CWE belongs to
def get_categories(cwe_id):
    categories = []
    for category in cwe_categories:
        if cwe_id in category.get('CWEs', []):
            categories.append({
                'category_id': category.get('Category_ID', 'Unknown'),
                'category_name': category.get('Category_Name', 'Unknown')
            })
    return categories

# Function to fetch CVEs related to a CWE
def get_cves_for_cwe(cwe_id):
    cves = []
    for cve_id, cve_info in cve_data.items():
        if cwe_id in cve_info.get('cwes', []):
            cves.append({
                'cve_id': cve_id,
                'description': cve_info.get('description', 'No description available.')
            })
    return cves

# Function to fetch mitigation details for techniques
def get_mitigation_details(technique_id):
    logging.info(f'Fetching mitigation details for technique: {technique_id}')
    mitigation_ids = []
    
    # Iterate over items in technique_mitigation_technique_mapping
    for item in technique_mitigation_technique_mapping:
        if item["technique_id"] == technique_id:
            mitigation_ids.append(item["technique_mitigation_id"])
    
    mitigations = []
    for mitigation_item in technique_mitigations:
        if mitigation_item['original_id'] in mitigation_ids:
            mitigations.append({
                'mitigation_id': mitigation_item['original_id'],
                'mitigation_name': mitigation_item.get('name', 'No name available.'),
                'mitigation_description': mitigation_item.get('description', 'No description available.')
            })
    
    return mitigations

# Function to generate dataset for BART
def generate_bart_dataset():
    dataset = []
    skipped_items = []
    
    for technique_id in technique_name_map:
        technique_info = get_technique_details(technique_id)
        if not technique_info or not technique_info['mitigations']:
            # Skip items with no mitigations
            skipped_items.append({'technique_id': technique_id, 'details': technique_info})
            continue
        
        # Fetch CAPECs related to the technique
        if technique_id in capec_techniques_map:
            for capec_id in capec_techniques_map[technique_id]:
                capec_info = get_capec_details(capec_id)
                if not capec_info:
                    continue
                
                # Fetch CWEs related to the CAPEC
                cwes = get_cwes_for_capec(capec_id)
                for cwe in cwes:
                    cwe_id = cwe['cwe_id']
                    
                    # Fetch CVEs related to the CWE
                    cves = get_cves_for_cwe(cwe_id)
                    for cve in cves:
                        input_text = (f"Technique Description: {technique_info['description']} - "
                                      f"Attack: {capec_info['capec_name']} - "
                                      f"Weakness: {cwe['cwe_name']} - Weakness Description: {cwe['description']} - "
                                      f"Vulnerability Description: {cve['description']}"
                                      f"Generate Title, Overview, and Recommendations:"  
                                    )
                        output = {
                            'title': technique_info['technique_name'],
                            'overview': capec_info['description'],
                            'mitigations': [f"{mitigation['mitigation_description']}-"for mitigation in technique_info['mitigations']]
                        }
                        dataset.append({'input': input_text, 'output': output})
    
    return dataset, skipped_items

# Main Function
if __name__ == '__main__':
    dataset, skipped_items = generate_bart_dataset()
    
    # Save the dataset to a JSON file
    with open('technique_to_cve_dataset.json', 'w') as outfile:
        json.dump(dataset, outfile, indent=2)
        logging.info('Technique to CVE dataset saved to technique_to_cve_dataset.json successfully.')
    
    # Save skipped items to a separate file
    with open('skipped_items.json', 'w') as outfile:
        json.dump(skipped_items, outfile, indent=2)
        logging.info('Skipped items saved to skipped_items.json successfully.')
