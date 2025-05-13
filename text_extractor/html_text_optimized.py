from pathlib import Path
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import os

def find_html_files(root_path: Path) -> list[Path]:
    return list(root_path.rglob("*.html")) + list(root_path.rglob("*.htm"))

def extract_chunks_from_html(html_content: str) -> list[str]:
    soup = BeautifulSoup(html_content, 'html.parser')
    chunks = []
    chunk_parts = []

    for element in soup.find_all(['h2', 'p', 'table']):
        if element.name == 'h2':
            if chunk_parts:
                chunks.append(' '.join(chunk_parts).replace('\n', ' ').strip())
                chunk_parts = []
            chunk_parts.append(element.get_text())
        elif element.name == 'p':
            chunk_parts.append(element.get_text())
        elif element.name == 'table':
            table_data = ' '.join(
                ' '.join(cell.get_text() for cell in row.find_all(['td', 'th']))
                for row in element.find_all('tr')
            )
            chunk_parts.append(table_data)

    if chunk_parts:
        chunks.append(' '.join(chunk_parts).replace('\n', ' ').strip())

    return chunks

def process_single_file(file_path: Path, input_dir: Path, output_dir: Path, extension: str):
    try:
        with file_path.open('r', encoding='utf-8') as file:
            html_content = file.read()

        chunks = extract_chunks_from_html(html_content)

        relative_path = file_path.relative_to(input_dir).with_suffix(extension)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w', encoding='utf-8') as out_file:
            out_file.write('\n\n'.join(chunks))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def process_html_files_parallel(input_dir: Path, output_dir: Path, extension: str = ".txt", max_workers: int = os.cpu_count()):
    html_files = find_html_files(input_dir)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file_path in html_files:
            executor.submit(process_single_file, file_path, input_dir, output_dir, extension)

# === Run the parallel processing ===
input_path = Path("C:/sj/data/html")
output_path = Path("C:/sj/data/output")
process_html_files_parallel(input_path, output_path)
