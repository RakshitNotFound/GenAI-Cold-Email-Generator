import pandas as pd
import chromadb
import uuid
import os


class Portfolio:
    def __init__(self, file_path=None):
        # Get the absolute path to the resource directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if file_path is None:
            file_path = os.path.join(current_dir, "resource", "my_portfolio.csv")
        
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Portfolio CSV file not found at: {file_path}")
            
        self.file_path = file_path
        self.data = pd.read_csv(self.file_path)
        self.chroma_client = chromadb.PersistentClient('vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(documents=row["Techstack"],
                                    metadatas={"links": row["Links"]},
                                    ids=[str(uuid.uuid4())])

    def query_links(self, skills):
        # Validate input
        if not skills:
            return []
        
        # Convert skills to list if it's a string
        if isinstance(skills, str):
            skills = [skills]
        
        # Ensure skills is not empty and contains valid text
        if not all(isinstance(s, str) and s.strip() for s in skills):
            raise ValueError("Skills must be non-empty strings")

        try:
            # Query the collection
            results = self.collection.query(
                query_texts=skills,
                n_results=2
            )
            
            # Extract and return metadata (links)
            metadatas = results.get('metadatas', [])
            return metadatas if metadatas else []
            
        except Exception as e:
            print(f"Error querying collection: {str(e)}")
            return []
