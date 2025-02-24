
import yaml 

def parse_log_labels(file_path):
    log_types = {}
    current_app = None
    current_type = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Detect application type
            if line.startswith('###'):
                current_app = line.replace('###', '').strip()
                log_types[current_app] = {}
                continue
                
            # Detect failure type
            if line.endswith(':'):
                current_type = line[:-1]  # Remove the colon
                log_types[current_app][current_type] = []
                continue
                
            # Add application ID if line starts with +
            if line.startswith('+'):
                app_id = line.replace('+', '').strip()
                log_types[current_app][current_type].append(app_id)
                
    return log_types


def load_yaml_config(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == '__main__':
    # Example usage
    file_path = "LLMs/data/Hadoop/abnormal_label.txt"
    log_types = parse_log_labels(file_path)
    print(log_types) 
    # Print results
    for app, failure_types in log_types.items():
        print(f"\n{app}:")
        for failure_type, apps in failure_types.items():
            print(f"  {failure_type}: {len(apps)} applications")
