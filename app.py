import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from groq import Groq
import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv() 

class NotebookDocumenter:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def extract_cells(self, notebook_path: str) -> List[Dict]:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        return [{'type': cell.cell_type, 'content': cell.source} for cell in nb.cells]
    
    def get_notebook_overview(self, cells: List[Dict]) -> str:
        """Generate an overview of what the notebook does"""
        all_code = "\n".join([cell['content'] for cell in cells if cell['type'] == 'code'])
        
        prompt = """Understand this Jupyter notebook code and provide:
        1. Start with generating a title describing the notebook's purpose , and dont wirte "Title:"
        2. A brief summary (2-3 sentences) about what the notebook does, without extra details.and dont wirte "Summary"

        Keep it concise and technical. Write as a data scientist would document their work.
        No placeholder text or generic descriptions.

        Code:
        ```python
        {code}
        ```""".format(code=all_code)

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "# Jupyter Notebook Documentation\n\n*Error generating overview*"

    def generate_cell_doc(self, cell_content: str, full_context: str) -> str:
        """Generate natural documentation for a code cell"""
        prompt = f"""As an expert data scientist documenting your work, briefly explain what this code does in the context of the notebook's workflow. 
        Write naturally as if documenting your own work, not as an AI, 
        if it is null cell dont generate anything and generate text based on complexity and length of the code present in the cell.

        Full notebook context:
        ```python
        {full_context}
        ```

        Current code:
        ```python
        {cell_content}
        ```

       Requirements:
       - Write in third person technical documentation style
       - No "This code..." or "Here we..." phrases
       - Be extremely concise - even one phrase is fine if it captures the point
       - Focus only on meaningful operations
       - Mention variable names only if crucial
       - Skip obvious operations

       Example good responses:
       - "Normalizes features using StandardScaler"
       - "Extracts timestamp from log entries for temporal analysis"
       - "Merges preprocessed datasets on user_id"
       - "Custom function to handle missing GPS coordinates"

       Example bad responses:
       - "This code performs..."
       - "Here we can see..."
       - "This cell is about..."
       - Any obvious/redundant explanations"""

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return "*Error generating documentation*"

    def create_documented_notebook(self, input_path: str, output_path: str):
        """Create documented notebook with natural writing style"""
        cells = self.extract_cells(input_path)
        new_nb = new_notebook()
        
        # Get notebook overview
        overview = self.get_notebook_overview(cells)
        new_nb.cells.append(new_markdown_cell(overview))
        
        # Full context for reference
        full_context = "\n\n".join([
            cell['content'] for cell in cells if cell['type'] == 'code'
        ])
        
        # Process each cell
        for cell in cells:
            if cell['type'] == 'code' and cell['content'].strip():
                doc = self.generate_cell_doc(cell['content'], full_context)
                if doc:
                    new_nb.cells.append(new_markdown_cell(doc))
                new_nb.cells.append(new_code_cell(cell['content']))
            elif cell['type'] == 'markdown':
                new_nb.cells.append(new_markdown_cell(cell['content']))
        
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(new_nb, f)
        print(f"Documentation complete. Output saved to: {output_path}")

def main():
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise ValueError("Please set GROQ_API_KEY environment variable")
    
    documenter = NotebookDocumenter(api_key)
    documenter.create_documented_notebook("test2.ipynb", "documented_notebook7.ipynb")

if __name__ == "__main__":
    main()
