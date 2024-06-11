def read_file(file):
    try:
        with open(file, 'r') as file:
            contents = file.read()
            return contents
    except FileNotFoundError:
        print(f"Error: File '{file}' not found.")
        return None


file = "text.txt"
file_cont = read_file(file)
if file_cont is not None:
    print("File contents:")
    print(file_cont)
