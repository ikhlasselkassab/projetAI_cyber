import re
from utils.technique_infos import get_technique_details
from utils.attack_infos import get_capec_details
from utils.attack_weaknesses import get_cwes_for_capec
from utils.weakness_vulns import get_cves_for_cwe
from utils.attack_parent_finder import get_final_attack

def generate_dataset_from_capec(
        capec_names, capec_descriptions, capec_cwe_mapping, 
        cwe_names, cwe_descriptions, cwe_categories, cve_data, 
        technique_name_map, technique_descriptions, technique_mitigation_technique_mapping, 
        technique_mitigations, capec_techniques_map, capec_from_xml, softwares, software_technique
        ):
    """
    Generate a BART dataset from CAPEC information.

    Args:
        capec_names (dict): Mapping of CAPEC IDs to names.
        capec_descriptions (dict): Mapping of CAPEC IDs to descriptions.
        capec_cwe_mapping (dict): Mapping of CAPEC IDs to CWEs.
        cwe_names (dict): Mapping of CWE IDs to names.
        cwe_descriptions (dict): Mapping of CWE IDs to descriptions.
        cwe_categories (dict): Mapping of CWE IDs to categories.
        cve_data (dict): Mapping of CVE IDs to CVE information.
        technique_name_map (dict): Mapping of technique IDs to names.
        technique_descriptions (dict): Mapping of technique IDs to descriptions.
        technique_mitigation_technique_mapping (list): List of mappings between techniques and mitigations.
        technique_mitigations (list): List of mitigation details.
        capec_techniques_map (dict): Mapping of CAPEC IDs to techniques.
        capec_from_xml (dict): Hierarchical relationships of CAPECs.
        software (list) : List of software details used by attackers.
        software_technique (list) : List of mapping software to technique.

    Returns:
        list: A list of dictionaries with BART input-output pairs.
        list: A list of dictionaries with BART input-output pairs that don't have mitigations.
        list: A list of CAPEC IDs for which the data was skipped.
    """
    dataset = []
    skipped_dataset = []
    skipped_items = []
    
    # Iterate over all CAPECs
    for capec_id in capec_names:
        capec_info = get_capec_details(capec_id, capec_names, capec_descriptions)
        if not capec_info:
            skipped_items.append(capec_id)
            continue
        
        # Fetch CWEs related to the CAPEC
        cwes = get_cwes_for_capec(capec_id, capec_cwe_mapping, cwe_names, cwe_descriptions, cwe_categories)
        if not cwes:
            skipped_items.append(capec_id)
            continue
        
        # Collect mitigations across all related techniques
        technique_ids = capec_techniques_map.get(capec_id, [])
        all_mitigations = set()
        for technique_id in technique_ids:
            technique_info = get_technique_details(technique_id, technique_name_map, technique_descriptions, technique_mitigation_technique_mapping, technique_mitigations, softwares, software_technique)
            
            if technique_info and technique_info['mitigations']:
                for mitigation in technique_info['mitigations']:
                    all_mitigations.add(mitigation['mitigation_name'])
        
        # Fetch the final attack in the hierarchy
        final_attack_info = get_final_attack(capec_id, capec_names, capec_descriptions, capec_from_xml)

        if not final_attack_info:
            skipped_items.append(capec_id)
            continue

        for mit in final_attack_info.get('mitigations', []):
            if mit:  # Optional: check if 'mit' is not empty or None
                all_mitigations.add(mit)
        
        # Generate entries for each CWE and CVE
        for cwe in cwes:
            cwe_id = cwe['cwe_id']
            cves = get_cves_for_cwe(cwe_id, cve_data)
            if not cves:
                skipped_items.append(capec_id)
                continue
            
            # Gather software details for the attack
            software_ids = [entry['software_id'] for entry in software_technique if entry['technique_id'] in technique_ids]
            software_details = [{'name': software['name'], 'type': software['type'], 'description': re.sub(r'\[.*?\]\(.*?\)', '', software['description']).strip()} for software in softwares if software['original_id'] in software_ids]
            software_text = ', '.join([f"{s['name']} ({s['type']})" for s in software_details]) if software_details else "No software details available."
            
            # Handle cases where final attack description might be empty or contain invalid data
            for cve in cves:
                if all_mitigations and capec_info['description'] != '' and cwe['description'] != '' and final_attack_info['description'] != '':
                    input_text = (f"Attack: {capec_info['capec_name']} - Attack Description: {capec_info['description']}\n"
                                f"Weakness: {cwe['cwe_name']} - Weakness Description: {cwe['description']} - "
                                f"Weakness Categories: {', '.join([category['category_name'] for category in cwe['categories']])} - "
                                f"Vulnerability Description: {cve['description']} - "
                                f"Software Used: {software_text} - "
                                )
                    output = {
                        'title': final_attack_info['capec_name'],
                        'overview': final_attack_info['description'],
                        'recommendations': ' '.join(f'{mit}-' for mit in list(all_mitigations))
                    }
                    dataset.append({'input': input_text, 'output': output})
                else:
                    input_text = (f"Attack: {capec_info['capec_name']} - Attack Description: {capec_info['description']}\n"
                                f"Weakness: {cwe['cwe_name']} - Weakness Description: {cwe['description']} - "
                                f"Weakness Categories: {', '.join([category['category_name'] for category in cwe['categories']])} - "
                                f"Vulnerability Description: {cve['description']} - "
                                f"Software Used: {software_text} - "
                                )
                    output = {
                        'title': final_attack_info['capec_name'],
                        'overview': final_attack_info['description'],
                        'recommendations': ' '.join(f'{mit}-' for mit in list(all_mitigations))
                    }
                    skipped_dataset.append({'input': input_text, 'output': output})
    
    return dataset, skipped_dataset, skipped_items