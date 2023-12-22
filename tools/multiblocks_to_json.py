import sys
import re
import json
import argparse


def parse_log_data(data):
    sections = re.split(r'bye bye :+\)', data)
    result = []

    for section in sections[:-1]:  # Ignore the last empty section
        m_match = re.search(r'm = (\d+)', section)
        m_value = int(m_match.group(1))

        cpu_time_match = re.search(r'CPU time: (\d+\.\d+) ms', section)
        cpu_time_ms = float(cpu_time_match.group(1))

        gpu_methods = []
        method_matches = re.finditer(r'Method: (\w+)\n(?:Total Time: (\d+\.?\d*) ms\nKernel Time: (\d+\.?\d*) ms\n)+MSE: (\d+\.?\d*)', section)
        for match in method_matches:
            method_name = match.group(1)
            total_time_values = list(map(float, re.findall(r'Total Time: (\d+\.?\d*) ms', match.group(0))))
            kernel_time_values = list(map(float, re.findall(r'Kernel Time: (\d+\.?\d*) ms', match.group(0))))
            mse = float(match.group(4))

            method_data = {
                "method": method_name,
                "total_time_ms": total_time_values,
                "kernel_time_ms": kernel_time_values,
                "MSE": mse
            }
            gpu_methods.append(method_data)

        entry = {
            "m": m_value,
            "cpu_time_ms": cpu_time_ms,
            "gpu_methods": gpu_methods
        }
        result.append(entry)

    return result


def parse_from_arg(input_filename, parser=None):
    log_data = None
    if input_filename is not None:
        with open(input_filename, "r") as f:
            log_data = f.read()
    if log_data is None:
        if not sys.stdin.isatty():
            log_data = sys.stdin.read()
    if log_data is None or len(log_data) == 0:
        if parser is not None:
            parser.print_help()
        exit(0)

    parsed_data = parse_log_data(log_data)
    return parsed_data

def main():

    parser = argparse.ArgumentParser(description='Convert log data to JSON format.')
    parser.add_argument('-o', '--output', help='Output JSON file')
    parser.add_argument('-i', '--input', help='Input log file')
    args = parser.parse_args()
    parsed_data = parse_from_arg(args.input)


    if args.output:
        with open(args.output, 'w') as json_file:
            json.dump(parsed_data, json_file, indent=3)
        print("Saved as \"%s.json\""%(args.output))
    else:
        print(json.dumps(parsed_data, indent=3))

if __name__ == "__main__":
    main()

