from typing import Dict, Any, List
import re

def parse_benchmark_data(log: str) -> List[Dict[str, str]]:
    lines = log.strip().split('\n')
    headers = lines[0].split('|')
    headers = [header.strip() for header in headers if header.strip()]
    data_lines = lines[2:]  # Skip the header and dashes rows

    parsed_data = []
    for line in data_lines:
        values = line.split('|')
        values = [value.strip() for value in values if value.strip()]
        if len(values) == len(headers):  # Ensure the number of values matches the number of headers
            obj = {headers[i]: values[i] for i in range(len(headers))}
            if all(obj.values()) and 'build:' not in obj.get('model', ''):
                parsed_data.append(obj)
    return parsed_data

def clean_key(key: str) -> str:
    """Remove type information from key names by taking first part before space"""
    return key.split()[0] if ' ' in key else key

def parse_model_loader_section(lines: List[str]) -> Dict[str, Any]:
    data = {
        "general": {},
        "llama": {
            "attention": {},
            "rope": {}
        },
        "tokenizer": {
            "ggml": {}
        }
    }
    
    for line in lines:
        if not line.startswith("llama_model_loader:"): 
            continue
            
        # Split after prefix and number
        try:
            # Example: "llama_model_loader: - kv   0: general.architecture str = llama"
            prefix, content = line.split(": ", 1)  # Split on first ": "
            _, content = content.split(":", 1)     # Remove "- kv NUMBER:"
            key, value = content.split("=", 1)     # Split on "="
            
            # Clean and split key parts
            key_parts = key.strip().split(".")     # Split path components
            final_key = clean_key(key_parts[-1])   # Clean type from last part
            value = value.strip()                  # Clean value
            
            # Map to correct section
            if "general" in key_parts:
                data["general"][final_key] = value
            elif "llama" in key_parts:
                if "attention" in key_parts:
                    data["llama"]["attention"][final_key] = value
                elif "rope" in key_parts:
                    data["llama"]["rope"][final_key] = value
                else:
                    data["llama"][final_key] = value
            elif "tokenizer" in key_parts:
                data["tokenizer"]["ggml"][final_key] = value
                
        except ValueError:
            continue
            
    return data

def parse_meta_section(lines: List[str]) -> Dict[str, str]:
    data = {}
    for line in lines:
        if not line.startswith("llm_load_print_meta:"):
            continue
        parts = line.split('=', 1)
        if len(parts) != 2:
            continue
        key = parts[0].replace('llm_load_print_meta:', '').strip()
        value = parts[1].strip()
        data[re.sub(r'\s+', '', key)] = value
    return data

def parse_system_info(line: str) -> Dict[str, Any]:
    if not line.startswith("system_info:"):
        return {}
    info = {}
    parts = line.replace("system_info:", "").split("|")
    for part in parts:
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            info[key.strip()] = int(value) if value.strip().isdigit() else value.strip()
    return info

def parse_perf_context(lines: List[str]) -> Dict[str, str]:
    data = {}
    for line in lines:
        if not line.startswith("llama_perf_context_print:"):
            continue
        parts = line.split('=', 1)
        if len(parts) != 2:
            continue
        key = parts[0].replace('llama_perf_context_print:', '').strip()
        value = parts[1].strip()
        data[re.sub(r'\s+', '', key)] = value
    return data

def parse_perplexity_data(log: str) -> Dict[str, Any]:
    """Parse llama.cpp log output into structured data"""
    lines = log.strip().split('\n')
    
    data = {
        "llama_model_loader": parse_model_loader_section(lines),
        "llm_load_print_meta": parse_meta_section(lines),
        "system_info": {},
        "llama_perf_context_print": parse_perf_context(lines),
        "final_estimate": {}
    }

    for line in lines:
        if 'Final estimate:' in line:  # More lenient matching
            try:
                # Extract numeric values
                parts = line.split('Final estimate: PPL = ')[1].split(' +/- ')
                if len(parts) == 2:
                    data['final_estimate'] = {
                        'ppl': float(parts[0].strip()),
                        'uncertainty': float(parts[1].strip())
                    }
                    print(f"Found final estimate: {data['final_estimate']}")  # Debug print
            except (IndexError, ValueError) as e:
                print(f"Failed to parse final estimate from: {line}")
                print(f"Error: {e}")
        elif line.startswith('system_info:'):
            data['system_info'] = parse_system_info(line)

    return data

def is_safe_cli_alias(alias: str) -> bool:
    """
    Validates that a CLI alias contains only safe characters.
    Allowed: alphanumeric, hyphens, underscores.
    """
    # Regex to match only alphanumeric characters, hyphens, and underscores.
    # Ensures the alias is simple and safe for use in logs and as a key.
    return re.match(r'^[a-zA-Z0-9_-]+$', alias) is not None