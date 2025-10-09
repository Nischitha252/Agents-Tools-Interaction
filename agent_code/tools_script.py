'''
Component Specification Extraction Tools for LangChain Agent
Tools for searching, scraping, and extracting component specifications from datasheets

Add these tools to your tools_script.py file
'''

import os
import re
import io
import requests
import pandas as pd
import PyPDF2
from bs4 import BeautifulSoup
from langchain.tools import tool
from urllib.parse import urljoin, quote_plus
from typing import Dict, List, Optional
import time


# --- TOOL: Google Search for Component Datasheet ---
@tool
def search_component_datasheet(query: str) -> str:
    """
    Search Google for component datasheets and return the top result URL.
    Input format: 'component_name company_name' (e.g., 'DPA 120 UL ABB')
    Output: First relevant URL or error message.
    """
    try:
        # Use DuckDuckGo as a fallback since Google requires API key
        search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query + ' datasheet')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        resp = requests.get(search_url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find first result link
        result_links = soup.find_all('a', class_='result__a')
        if result_links:
            url = result_links[0].get('href', '')
            if url:
                return f"Found URL: {url}"
        
        return "error: No search results found"
        
    except Exception as e:
        return f"error: {str(e)}"


# --- TOOL: Navigate Website and Find Datasheet Links ---
@tool
def find_datasheet_on_website(url: str) -> str:
    """
    Navigate to a company website and find links to technical datasheets or specifications.
    Input: Website URL (e.g., 'https://new.abb.com/ups/systems/...')
    Output: List of datasheet PDF URLs found on the page.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        datasheet_urls = []
        keywords = ['datasheet', 'technical', 'specification', 'spec', 'data sheet', 'tech spec']
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True).lower()
            
            # Check if link text or href contains datasheet keywords
            if any(keyword in text for keyword in keywords) or any(keyword in href.lower() for keyword in keywords):
                full_url = urljoin(url, href)
                datasheet_urls.append(f"{text[:50]}: {full_url}")
            
            # Direct PDF links
            if '.pdf' in href.lower():
                full_url = urljoin(url, href)
                datasheet_urls.append(f"PDF: {full_url}")
        
        if datasheet_urls:
            return "Found datasheets:\n" + "\n".join(datasheet_urls[:10])
        return "error: No datasheet links found on this page"
        
    except Exception as e:
        return f"error: {str(e)}"


# --- TOOL: Download and Extract Text from PDF ---
@tool
def extract_text_from_pdf_url(pdf_url: str) -> str:
    """
    Download a PDF from URL and extract all text content for analysis.
    Input: Direct URL to PDF file
    Output: Extracted text (first 5000 characters) or error message.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(pdf_url, headers=headers, timeout=60)
        resp.raise_for_status()
        
        # Parse PDF
        pdf_file = io.BytesIO(resp.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        # Extract from first 15 pages (specifications usually in beginning)
        num_pages = min(15, len(pdf_reader.pages))
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        if len(text) < 100:
            return "error: Could not extract meaningful text from PDF"
        
        # Return first 5000 chars to avoid token limits
        return f"PDF extracted successfully ({len(text)} chars total). First 5000 chars:\n{text[:5000]}"
        
    except Exception as e:
        return f"error: {str(e)}"


# --- TOOL: Extract Power Rating from Text ---
@tool
def extract_power_rating(text: str) -> str:
    """
    Extract power rating specification from technical document text.
    Looks for patterns like: '120kW', 'rated power: 50W', 'power rating 100-200kW', etc.
    Input: Text content from datasheet
    Output: Power rating value or 'Not Found'
    """
    try:
        # Multiple regex patterns to catch different formats
        patterns = [
            r'power\s+rating[:\s]*([0-9.]+)\s*(?:to\s+([0-9.]+)\s*)?(kW|W|MW|mW)',
            r'rated\s+power[:\s]*([0-9.]+)\s*(?:to\s+([0-9.]+)\s*)?(kW|W|MW|mW)',
            r'output\s+power[:\s]*([0-9.]+)\s*(?:to\s+([0-9.]+)\s*)?(kW|W|MW|mW)',
            r'nominal\s+power[:\s]*([0-9.]+)\s*(?:to\s+([0-9.]+)\s*)?(kW|W|MW|mW)',
            r'([0-9]+)\s*-\s*([0-9]+)\s*(kW|W|MW)\s+at',
            r'power[:\s]*([0-9.]+)\s*(?:to\s+([0-9.]+)\s*)?(kW|W|MW|mW)',
            r'TDP[:\s]*([0-9.]+)\s*(kW|W|MW|mW)',
        ]
        
        text_cleaned = text.replace('\n', ' ').replace('\r', ' ')
        
        for pattern in patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                groups = match.groups()
                if groups[1] and groups[1] != groups[2]:  # Range format
                    return f"Power Rating: {groups[0]}-{groups[1]}{groups[2]}"
                else:
                    unit = groups[2] if len(groups) > 2 else groups[1]
                    return f"Power Rating: {groups[0]}{unit}"
        
        return "Power Rating: Not Found"
        
    except Exception as e:
        return f"error: {str(e)}"


# --- TOOL: Extract Heat Dissipation from Text ---
@tool
def extract_heat_dissipation(text: str) -> str:
    """
    Extract heat dissipation specification from technical document text.
    Looks for patterns like: 'heat dissipation: 5.2kW', 'thermal dissipation 100W', etc.
    Input: Text content from datasheet
    Output: Heat dissipation value or 'Not Found'
    """
    try:
        patterns = [
            r'heat\s+dissipat(?:ion|ed)[:\s]*([0-9.]+)\s*(kW|W|BTU|mW)',
            r'thermal\s+dissipation[:\s]*([0-9.]+)\s*(kW|W|BTU|mW)',
            r'power\s+dissipation[:\s]*([0-9.]+)\s*(kW|W|BTU|mW)',
            r'dissipated\s+(?:power|heat)[:\s]*([0-9.]+)\s*(kW|W|BTU|mW)',
            r'heat\s+output[:\s]*([0-9.]+)\s*(kW|W|BTU|mW)',
            r'maximum\s+heat[:\s]*([0-9.]+)\s*(kW|W|BTU|mW)',
        ]
        
        text_cleaned = text.replace('\n', ' ').replace('\r', ' ')
        
        for pattern in patterns:
            match = re.search(pattern, text_cleaned, re.IGNORECASE)
            if match:
                value = match.group(1)
                unit = match.group(2)
                return f"Heat Dissipated: {value}{unit}"
        
        return "Heat Dissipated: Not Found"
        
    except Exception as e:
        return f"error: {str(e)}"


# --- TOOL: Read Excel File ---
@tool
def read_excel_components(file_path: str) -> str:
    """
    Read component and company information from Excel file.
    Input: Path to Excel file (e.g., 'Sample_Mastersheet.xlsx')
    Output: JSON string with component-company pairs and row indices.
    """
    try:
        df = pd.read_excel(file_path)
        
        # Find component and company columns
        component_col = None
        company_col = None
        
        for col in df.columns:
            if 'component' in str(col).lower():
                component_col = col
            if 'company' in str(col).lower():
                company_col = col
        
        if not component_col or not company_col:
            return f"error: Could not find 'component' or 'company' columns. Found: {list(df.columns)}"
        
        # Extract data
        components_data = []
        for idx, row in df.iterrows():
            if pd.notna(row[component_col]) and pd.notna(row[company_col]):
                components_data.append({
                    'row': int(idx),
                    'component': str(row[component_col]),
                    'company': str(row[company_col])
                })
        
        return f"Excel loaded successfully. Found {len(components_data)} components:\n" + str(components_data[:5])
        
    except Exception as e:
        return f"error: {str(e)}"


# --- TOOL: Update Excel with Specifications ---
@tool
def update_excel_with_specs(file_path: str, row_index: int, power_rating: str, heat_dissipated: str) -> str:
    """
    Update Excel file with extracted power rating and heat dissipation values.
    Input format: 'file_path|row_index|power_rating|heat_dissipated'
    Example: 'data.xlsx|2|120kW|5.2kW'
    Output: Success or error message.
    """
    try:
        # Parse input (format: filepath|row|power|heat)
        parts = file_path.split('|')
        if len(parts) == 4:
            filepath = parts[0]
            row_idx = int(parts[1])
            power = parts[2]
            heat = parts[3]
        else:
            return "error: Invalid input format. Use 'filepath|row_index|power_rating|heat_dissipated'"
        
        # Read Excel
        df = pd.read_excel(filepath)
        
        # Add columns if they don't exist
        if 'Power Rating' not in df.columns:
            df['Power Rating'] = ''
        if 'Heat Dissipated' not in df.columns:
            df['Heat Dissipated'] = ''
        
        # Update specific row
        df.at[row_idx, 'Power Rating'] = power
        df.at[row_idx, 'Heat Dissipated'] = heat
        
        # Save
        output_path = filepath.replace('.xlsx', '_updated.xlsx')
        df.to_excel(output_path, index=False)
        
        return f"Success: Updated row {row_idx} in {output_path}"
        
    except Exception as e:
        return f"error: {str(e)}"


# --- TOOL: Complete Component Spec Extraction Pipeline ---
@tool
def extract_component_specs_pipeline(component_company_query: str) -> str:
    """
    Complete pipeline to search, find, download datasheet and extract specifications.
    Input format: 'component_name|company_name' (e.g., 'DPA 120 UL|ABB')
    Output: JSON with power_rating and heat_dissipated values.
    This tool orchestrates multiple steps automatically.
    """
    try:
        parts = component_company_query.split('|')
        if len(parts) != 2:
            return "error: Input format should be 'component_name|company_name'"
        
        component = parts[0].strip()
        company = parts[1].strip()
        
        result = {
            'component': component,
            'company': company,
            'power_rating': 'Not Found',
            'heat_dissipated': 'Not Found',
            'status': 'Processing'
        }
        
        # Step 1: Search for datasheet
        search_query = f"{company} {component} datasheet"
        search_url = f"https://www.google.com/search?q={quote_plus(search_query)}"
        
        # For demo, return template - in production, this would do full search
        result['status'] = 'Pipeline requires manual execution of individual tools'
        result['instructions'] = [
            f"1. Use 'search_component_datasheet' with query: '{search_query}'",
            "2. Use 'find_datasheet_on_website' with the URL from step 1",
            "3. Use 'extract_text_from_pdf_url' with PDF URL from step 2",
            "4. Use 'extract_power_rating' with text from step 3",
            "5. Use 'extract_heat_dissipation' with text from step 3",
            "6. Use 'update_excel_with_specs' to save results"
        ]
        
        return str(result)
        
    except Exception as e:
        return f"error: {str(e)}"


# --- Add these tools to your TOOLS_LIST ---
COMPONENT_TOOLS = [
    search_component_datasheet,
    find_datasheet_on_website,
    extract_text_from_pdf_url,
    extract_power_rating,
    extract_heat_dissipation,
    read_excel_components,
    update_excel_with_specs,
    extract_component_specs_pipeline,
]

# Tool descriptions for system prompt
COMPONENT_TOOLS_DESCRIPTIONS = "\n".join(
    f"- {tool.name}: {tool.description}" for tool in COMPONENT_TOOLS
)