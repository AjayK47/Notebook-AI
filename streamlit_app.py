import streamlit as st
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from groq import Groq
import os
from typing import List, Dict
from dotenv import load_dotenv
import requests
import json
from io import BytesIO
import re

load_dotenv()

class NotebookDocumenter:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
    
    def extract_cells(self, notebook_content) -> List[Dict]:
        """Extract cells from notebook content"""
        if isinstance(notebook_content, str):
            nb = nbformat.reads(notebook_content, as_version=4)
        else:
            nb = notebook_content
        return [{'type': cell.cell_type, 'content': cell.source} for cell in nb.cells]
    
    def get_notebook_overview(self, cells: List[Dict]) -> str:
        """Generate an overview of what the notebook does"""
        all_code = "\n".join([cell['content'] for cell in cells if cell['type'] == 'code'])
        
        prompt = """Understand this Jupyter notebook code and provide:
        1. Start with generating a title describing the notebook's purpose , and dont write "Title:"
        2. A brief summary (2-3 sentences) about what the notebook does, without extra details.and dont write "Summary"

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
                temperature=0.7,
                max_tokens=300,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"# Jupyter Notebook Documentation\n\n*Error generating overview: {str(e)}*"
 
    def generate_cell_doc(self, cell_content: str, full_context: str) -> str:
        """Generate natural documentation for a code cell"""
        if not cell_content.strip():
            return ""
            
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
       - Skip obvious operations.

       Example good responses:
       - "Normalizes features using StandardScaler"
       - "Extracts timestamp from log entries for temporal analysis"
       - "Merges preprocessed datasets on user_id"
       - "Custom function to handle missing GPS coordinates"

       Example bad responses:
       - "This code performs..."
       - "Here we can see..."
       - "This cell is about..."
       - " The code .."
       - Any obvious/redundant explanations       
       """
        

        try:
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=100,
                top_p=1,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"*Error generating documentation: {str(e)}*"

    def create_documented_notebook(self, notebook_content) -> nbformat.notebooknode.NotebookNode:
        """Create documented notebook with natural writing style"""
        cells = self.extract_cells(notebook_content)
        new_nb = new_notebook()
        
        # Get notebook overview
        with st.spinner('Generating notebook overview...'):
            overview = self.get_notebook_overview(cells)
            new_nb.cells.append(new_markdown_cell(overview))
        
        # Full context for reference
        full_context = "\n\n".join([
            cell['content'] for cell in cells if cell['type'] == 'code'
        ])
        
        # Process each cell
        total_cells = len(cells)
        progress_bar = st.progress(0)
        
        for idx, cell in enumerate(cells):
            if cell['type'] == 'code' and cell['content'].strip():
                doc = self.generate_cell_doc(cell['content'], full_context)
                if doc:
                    new_nb.cells.append(new_markdown_cell(doc))
                new_nb.cells.append(new_code_cell(cell['content']))
            elif cell['type'] == 'markdown':
                new_nb.cells.append(new_markdown_cell(cell['content']))
                
            progress_bar.progress((idx + 1) / total_cells)
        
        return new_nb

def download_colab_notebook(colab_url: str) -> dict:
    """Download notebook from Google Colab URL"""
    # Extract the file ID from the Colab URL using more flexible pattern matching
    patterns = [
        r'/d/([a-zA-Z0-9-_]+)',  # Matches /d/xxx format
        r'drive/([a-zA-Z0-9-_]+)',  # Matches drive/xxx format
        r'\/([a-zA-Z0-9-_]{20,})',  # Matches any long alphanumeric string
    ]
    
    file_id = None
    for pattern in patterns:
        match = re.search(pattern, colab_url)
        if match:
            file_id = match.group(1)
            break
    
    if not file_id:
        raise ValueError("Could not extract file ID from the provided Colab URL")
    
    # Try different download URL formats
    download_urls = [
        f"https://drive.google.com/uc?id={file_id}",
        f"https://drive.google.com/uc?export=download&id={file_id}"
    ]
    
    for download_url in download_urls:
        try:
            response = requests.get(download_url)
            if response.status_code == 200:
                try:
                    # Try to parse the notebook content
                    return nbformat.reads(response.content.decode('utf-8'), as_version=4)
                except:
                    continue
        except:
            continue
    
    # If we get here, none of the attempts worked
    raise ValueError("""Unable to download the notebook. This might be because:
    1. The notebook isn't publicly shared
    2. The URL format is not supported
    3. Google Drive access restrictions
    
    Please make sure the notebook is publicly accessible and try again.""")

def main():
    st.set_page_config(page_title="Notebook Documenter", layout="wide")
    
    st.title("📚 Jupyter Notebook Documenter")
    st.write("""
    This app automatically generates natural documentation for Jupyter notebooks. 
    Upload a notebook file or provide a Google Colab link to get started.
    """)

    # Add instructions for Colab sharing
    with st.expander("📋 How to share your Colab notebook"):
        st.write("""
        1. Open your Colab notebook
        2. Click 'Share' in the top-right corner
        3. Click 'Anyone with the link'
        4. Make sure 'Viewer' access is selected
        5. Click 'Copy link'
        6. Paste the link here
        
        The link should look something like:
        `https://colab.research.google.com/drive/1abc...xyz`
        """)

    # API Key input
    api_key = st.text_input("Enter your Groq API Key:", type="password")
    if not api_key:
        st.warning("Please enter your Groq API Key to continue")
        return

    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Upload Notebook", "Google Colab Link"]
    )

    try:
        documenter = NotebookDocumenter(api_key)
        
        if input_method == "Upload Notebook":
            uploaded_file = st.file_uploader("Choose a notebook file", type=['ipynb'])
            if uploaded_file:
                notebook_content = nbformat.read(uploaded_file, as_version=4)
                process_notebook = True
            else:
                process_notebook = False
                
        else:  # Google Colab Link
            colab_url = st.text_input("Enter Google Colab URL:")
            if colab_url:
                try:
                    with st.spinner('Downloading notebook from Colab...'):
                        notebook_content = download_colab_notebook(colab_url)
                    process_notebook = True
                except Exception as e:
                    st.error(f"""Error downloading Colab notebook: {str(e)}
                    
                    Please ensure:
                    1. The URL is correct
                    2. The notebook is shared with 'Anyone with the link'
                    3. The sharing settings allow viewing""")
                    process_notebook = False
            else:
                process_notebook = False

        if process_notebook:
            if st.button("Generate Documentation"):
                try:
                    with st.spinner('Generating documentation...'):
                        documented_nb = documenter.create_documented_notebook(notebook_content)
                        
                        # Convert notebook to string for download
                        notebook_str = nbformat.writes(documented_nb)
                        
                        # Create download button
                        st.download_button(
                            label="Download Documented Notebook",
                            data=notebook_str,
                            file_name="documented_notebook.ipynb",
                            mime="application/x-ipynb+json"
                        )
                        
                        st.success("Documentation generated successfully!")
                        
                        # Display preview
                        st.subheader("Preview")
                        for cell in documented_nb.cells[:5]:  # Show first 5 cells
                            if cell.cell_type == 'markdown':
                                st.markdown(cell.source)
                            else:
                                st.code(cell.source, language='python')
                        
                        if len(documented_nb.cells) > 5:
                            st.info("... (Download the notebook to see full documentation)")
                        
                except Exception as e:
                    st.error(f"Error generating documentation: {str(e)}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
