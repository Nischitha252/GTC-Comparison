from pandas.io.excel._xlsxwriter import XlsxWriter
from flask import jsonify
import re
import io
import pandas as pd
import logging

from .Azure_Translator import Translator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_excel_with_formatting_local(df, language,sheet_name):
    """
  Creates an Excel file with formatting for a given DataFrame, applying bold and text wrapping.

  Args:
    df: The DataFrame to be formatted.
    sheet_name: The name of the worksheet to create.

  Returns:
    A byte string containing the formatted Excel file.
  """
    # Create a temporary buffer for the Excel file in memory
    logger.info("Excel Formatting has Started")
    translator = Translator()
    output=io.BytesIO()
    # Access the workbook for formatting
    writer= pd.ExcelWriter(output, engine='xlsxwriter')

    workbook = writer.book
    bold = workbook.add_format({'bold': True})

    # Create the worksheet (if it doesn't already exist)
    if sheet_name not in writer.sheets:
        worksheet = writer.book.add_worksheet("Output")
        # Set column widths for readability
        worksheet.set_column(0, 0, 40)
        worksheet.set_column(1, 6, 70)

        header_format = workbook.add_format(
            {
                "bold": True,
                "text_wrap": True,
                "valign": "top",
                "fg_color": "#ADD8E6",
                "border": 1,
                "align": "center",  # Center alignment added
            }
        )
        top_aligned_format= workbook.add_format({"text_wrap": True,
                                                 "valign": "top"})
        

    def write_formatted_cell(row, col, value):
    # Convert non-string types to string
        if not isinstance(value, str):
            if isinstance(value, list):
                value = ' '.join(map(str, value))  # Convert list to string
            else:
                value = str(value)  # Convert other types (e.g., float) to string
        
        # Split the value by the bold markers
        parts = re.split(r'(\*\*.*?\*\*)', value)
        formatted_parts = []
        bold = workbook.add_format({'bold': True})
        
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                formatted_parts.append(bold)
                formatted_parts.append(part[2:-2])  # Remove the asterisks
            elif part:  # Only add non-empty parts
                formatted_parts.append(part)
        
        if len(formatted_parts) < 2:
            worksheet.write(row, col, value, top_aligned_format)
        else:
            worksheet.write_rich_string(row, col, *formatted_parts, top_aligned_format)
    
    try:
        # Write the DataFrame to the Excel file, starting from row 1 to avoid the header
        df.to_excel(writer, sheet_name="Output", startrow=1, header=False, index=False)
        # Write headers with custom formatting
        for col_num, value in enumerate(df.columns.values):
            translated_value=translator.translate(value, language)
            worksheet.write(0, col_num , translated_value, header_format)

        # Iterate through cells and apply formatting
        for row_num in range(1, len(df) + 1):
            for col_num in range(1,len(df.columns)):
                write_formatted_cell(row_num, col_num, df.iloc[row_num - 1, col_num])
            # Top Alignment for the first column
            worksheet.write(row_num, 0, df.iloc[row_num - 1, 0],top_aligned_format)

        writer.close()
        logger.info("Excel Formatting has Completed")
    except AttributeError as e:
        return jsonify({'success': False, 'message': f'Error Formatting the excel file: {str(e)}'})

    return output.getvalue()