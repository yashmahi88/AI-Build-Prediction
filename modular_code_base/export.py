import os

# Output file name
OUTPUT_FILE = "all_python_files2.txt"

def collect_python_files(root_dir):
    current_script = os.path.abspath(__file__)
    collected_code = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.abspath(os.path.join(root, file))

                # Exclude this script itself
                if full_path == current_script:
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    collected_code.append(
                        f"\n\n{'=' * 80}\n"
                        f"FILE: {full_path}\n"
                        f"{'=' * 80}\n\n"
                        f"{content}"
                    )

                except Exception as e:
                    collected_code.append(
                        f"\n\nERROR READING FILE: {full_path}\n{str(e)}\n"
                    )

    return "".join(collected_code)


if __name__ == "__main__":
    base_directory = os.getcwd()  # Change this if needed
    combined_text = collect_python_files(base_directory)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        out_file.write(combined_text)

    print(f"✅ All Python files exported to: {OUTPUT_FILE}")
