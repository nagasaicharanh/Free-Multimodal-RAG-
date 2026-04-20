"""
Table processor module for extracting and processing tables from PDFs.
Converts tables to markdown and generates summaries via LLM.
"""
from typing import List, Dict, Optional
import pandas as pd


class TableProcessor:
    """Process tables extracted from PDFs."""

    def __init__(self):
        pass

    def process_tables(self, tables: List[Dict]) -> List[Dict]:
        """
        Process extracted tables, enrich with metadata.
        
        Args:
            tables: List of table dicts from PDFParser
            
        Returns:
            Enriched table dictionaries ready for ChromaDB
        """
        processed = []
        for table_dict in tables:
            processed_table = {
                "page": table_dict["page"],
                "table_index": table_dict["table_index"],
                "markdown": table_dict["markdown"],
                "raw_data": table_dict["raw"],
                "num_rows": len(table_dict["raw"]) - 1,  # Exclude header
                "num_cols": len(table_dict["raw"][0]) if table_dict["raw"] else 0,
            }
            processed.append(processed_table)
        return processed

    def table_to_dataframe(self, table_data: List[List[str]]) -> Optional[pd.DataFrame]:
        """
        Convert raw table data to pandas DataFrame.
        
        Args:
            table_data: Raw table as list of lists
            
        Returns:
            pandas DataFrame or None if invalid
        """
        try:
            if not table_data or len(table_data) < 2:
                return None
            
            headers = table_data[0]
            rows = table_data[1:]
            
            df = pd.DataFrame(rows, columns=headers)
            return df
        except Exception as e:
            print(f"Error converting table to DataFrame: {e}")
            return None

    def get_table_summary_text(self, table_dict: Dict) -> str:
        """
        Generate a text summary for a table (for LLM processing).
        
        Args:
            table_dict: Processed table dictionary
            
        Returns:
            Text describing the table structure and content
        """
        summary = f"""
Table from Page {table_dict['page']}:
- Dimensions: {table_dict['num_rows']} rows × {table_dict['num_cols']} columns
- Content:
{table_dict['markdown']}
"""
        return summary.strip()


if __name__ == "__main__":
    # Example usage
    processor = TableProcessor()
    
    # Sample table data
    sample_table = [
        ["Name", "Age", "City"],
        ["Alice", "28", "New York"],
        ["Bob", "34", "San Francisco"],
        ["Charlie", "45", "Boston"],
    ]
    
    # Create table dict like PDFParser would return
    table_dict = {
        "page": 1,
        "table_index": 0,
        "markdown": processor.process_tables([{
            "page": 1,
            "table_index": 0,
            "markdown": "| Name | Age | City |\n|---|---|---|\n| Alice | 28 | New York |",
            "raw": sample_table,
        }])[0]["markdown"],
        "raw": sample_table,
    }
    
    # Process
    processed = processor.process_tables([table_dict])
    print(f"Processed: {processed[0]}")
