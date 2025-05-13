from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
from langchain.docstore.document import Document as LangchainDocument

# Constants
MARKDOWN_SEPARATORS = ["\n\n", "\n", " ", ""]
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
CHAR_LIMT = 200

class chunk_generator:
    def __init__(self):
        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )

    def process_files(self, files: list[Path]) -> list[str]:
        """Reads and wraps file contents into LangchainDocument objects."""
        contents = []
        combined_small_files = ''
        for file_path in files:
            if file_path.is_file():
                with file_path.open('r', encoding='utf-8') as file: 
                    content = file.read() 
                    if len(content) < CHAR_LIMT:
                        combined_small_files += content + "\n"
                    else:
                        if(len(combined_small_files) > 0):
                            combined_small_files += content
                            contents.append(combined_small_files) 
                            combined_small_files = '' 
                        else:
                            contents.append(content)
        if(len(combined_small_files) > 0):
            contents.append(combined_small_files)                       
        return contents

    def collect_documents(self, root_path: Path) -> list[LangchainDocument]:
        """Recursively collects and processes all documents from a root path."""
        all_documents = []
        for folder in [p for p in root_path.rglob("*") if p.is_dir()] + [root_path]:
            files = list(folder.glob("*"))
            documents = self.process_files(files)  
            for doc in documents:
                all_documents.append(LangchainDocument(page_content=doc))        
        return all_documents

    def create_chunks(self, path: str):
        root_path = Path(path)
        raw_documents = self.collect_documents(root_path)
        split_documents = self.text_splitter.split_documents(raw_documents)
        return split_documents








