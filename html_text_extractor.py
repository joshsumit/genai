
from pathlib import Path
from bs4 import BeautifulSoup

inputpath = "C:\\data\\html" 
output_directory = Path("C:\\data\\txt")  
extension = ".txt"

from pathlib import Path

def find_html_files(root_path: Path) -> list[Path]:
    return [p for p in root_path.rglob('*') if p.is_file() and p.suffix.lower() in {'.html', '.htm'}]

files = find_html_files(Path(inputpath))

for filename in files:
    with open(filename, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = []
    chunk = ''
    previous=''
    for element in soup.find_all(['h2', 'p', 'table']):
        if element.name == 'h2':
            if chunk != '':                
                chunks.append(chunk.replace('\n', ' '))
                chunk=''            
            chunk  += element.get_text() + ' '
        elif element.name == 'p':            
            #chunk.append({'type': 'text', 'content': element.get_text()})
            chunk  += element.get_text() + ' '           
        elif element.name == 'table':
            table_data = ''
            for row in element.find_all('tr'):
                row_data = [cell.get_text() for cell in row.find_all(['td', 'th'])]
                table_data +=  " ".join(row_data)
            #chunk.append({'type': 'table', 'content': table_data})
            chunk += table_data.replace('\n', '')
            #previous = 'table'

    if chunk != '':
        chunk = chunk.replace('\n', ' ').strip() + '\n'
        chunks.append(chunk)
    
    
    # Compute relative path and new output path
    relative_path = filename.relative_to(inputpath).with_suffix(extension)
    new_path = output_directory / relative_path

    # Ensure the output directory exists
    new_path.parent.mkdir(parents=True, exist_ok=True)


    # new_path = output_directory / (filename.stem + extension)

    with open(Path(new_path), 'w', encoding='utf-8') as file:
        # Write each item in the list to the file
        for item in chunks:
            file.write(f"{item}\n")

