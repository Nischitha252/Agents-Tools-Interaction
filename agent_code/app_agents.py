'''
Component Specification Extraction Agent
Integrated with LangChain agent framework for automated datasheet extraction

Created by: Nischitha M K

'''

import os
import pandas as pd
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

# Import all tools
from tools_script import COMPONENT_TOOLS, COMPONENT_TOOLS_DESCRIPTIONS


# Load environment variables
load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')


# --- Enhanced System Prompt with Component Extraction Instructions ---
system_prompt = f"""
You are an advanced AI assistant specialized in extracting component specifications from technical datasheets.

**Your Core Capabilities:**
1. General knowledge and information retrieval
2. **Component specification extraction from company websites and datasheets**
3. Web scraping and PDF document analysis
4. Excel file manipulation

**Available Tools:**
{COMPONENT_TOOLS}

**Component Specification Tools:**
{COMPONENT_TOOLS_DESCRIPTIONS}

**ReAct Framework - Use this reasoning pattern:**
Thought: (analyze what needs to be done)
Action: (select the appropriate tool)
Action Input: (provide input to the tool)
Observation: (result from the tool)
... (repeat as needed)
Final Answer: (conclude with the result)

**IMPORTANT INSTRUCTIONS FOR COMPONENT EXTRACTION:**

When asked to extract component specifications from a website or datasheet:

1. **Search Phase:**
   - Use 'search_component_datasheet' to find the company website
   - Extract the first result URL

2. **Navigation Phase:**
   - Use 'find_datasheet_on_website' with the URL to locate datasheet links
   - Look for links containing "technical specification", "datasheet", or "technical data"

3. **Extraction Phase:**
   - Use 'extract_text_from_pdf_url' to download and read the PDF
   - Parse the extracted text carefully

4. **Specification Identification:**
   - Use 'extract_power_rating' on the PDF text to find power specifications
   - Use 'extract_heat_dissipation' on the PDF text to find thermal specifications
   - Look for patterns like:
     * Power: "120kW", "20-60kW at 208V", "rated power: 100W"
     * Heat: "heat dissipation: 5.2kW", "thermal output: 100W"

5. **Excel Operations:**
   - Use 'read_excel_components' to load component data from Excel
   - Use 'update_excel_with_specs' to save extracted values

**Example Workflow:**
User: "Extract specs for DPA 120 UL from ABB"

Thought: I need to search for ABB DPA 120 UL datasheet
Action: search_component_datasheet
Action Input: DPA 120 UL ABB

Thought: I found the URL, now I need to find datasheet links
Action: find_datasheet_on_website
Action Input: https://new.abb.com/ups/systems/three-phase-ups/conceptpower-dpa-120-ul

Thought: I found datasheet PDF, let me extract text
Action: extract_text_from_pdf_url
Action Input: https://search.abb.com/library/Download.aspx?DocumentID=...

Thought: I have the text, let me extract power rating
Action: extract_power_rating
Action Input: [PDF text content]

Thought: Now extract heat dissipation
Action: extract_heat_dissipation
Action Input: [PDF text content]

Final Answer: 
Component: Conceptpower DPA 120 UL
Company: ABB
Power Rating: 20-120kW at 208V
Heat Dissipated: 5.2kW

**Error Handling:**
- If a tool returns "error:", try an alternative approach
- If no datasheet found after 3 attempts, respond with "Not Found"
- Always verify extracted values make sense (e.g., power should be > 0)

**For Excel Processing:**
When given an Excel file path, process each row sequentially:
1. Read all components using 'read_excel_components'
2. For each component-company pair:
   - Extract specifications using the workflow above
   - Update the Excel file using 'update_excel_with_specs'
3. Report progress and summary

You must not hallucinate values. If a specification is not found, explicitly state "Not Found".
"""


# --- Initialize LLM ---
chat_llm = ChatOpenAI(
    openai_api_key=OPENAI_KEY,
    model_name=OPENAI_MODEL,
    temperature=0.1,  # Lower temperature for more accurate extraction
)



# --- Initialize Agent ---
agent = initialize_agent(
    tools=COMPONENT_TOOLS,
    llm=chat_llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs={"system_message": system_prompt},
    max_iterations=15,  # More iterations for complex workflows
    max_execution_time=300,  # 5 minutes timeout
    early_stopping_method="generate",
    handle_parsing_errors=True,
    verbose=True,
)


# --- Helper Function for Batch Processing ---
def process_excel_file(file_path: str) -> str:
    """
    Process an entire Excel file and extract specifications for all components
    """
    try:
        df = pd.read_excel(file_path)
        
        # Find columns
        component_col = None
        company_col = None
        for col in df.columns:
            if 'component' in str(col).lower():
                component_col = col
            if 'company' in str(col).lower():
                company_col = col
        
        if not component_col or not company_col:
            return f"Error: Could not find required columns in {file_path}"
        
        # Add result columns
        if 'Power Rating' not in df.columns:
            df['Power Rating'] = ''
        if 'Heat Dissipated' not in df.columns:
            df['Heat Dissipated'] = ''
        
        results = []
        total = len(df)
        
        # Process each row
        for idx, row in df.iterrows():
            component = row[component_col]
            company = row[company_col]
            
            if pd.isna(component) or pd.isna(company):
                continue
            
            # Ask agent to extract specs
            query = f"Extract power rating and heat dissipation for component '{component}' from company '{company}'. Search their website, find the technical datasheet PDF, and extract these two specifications."
            
            try:
                result = agent.run(query)
                results.append(f"[{idx+1}/{total}] {component} | {company}: {result}")
                
                # Try to parse result and update Excel
                # (In production, you'd have more robust parsing)
                if "Power Rating:" in result and "Heat Dissipated:" in result:
                    power = result.split("Power Rating:")[1].split("Heat Dissipated:")[0].strip()
                    heat = result.split("Heat Dissipated:")[1].split("\n")[0].strip()
                    
                    df.at[idx, 'Power Rating'] = power
                    df.at[idx, 'Heat Dissipated'] = heat
                
            except Exception as e:
                results.append(f"[{idx+1}/{total}] Error processing {component}: {str(e)}")
        
        # Save updated Excel
        output_path = file_path.replace('.xlsx', '_output.xlsx')
        df.to_excel(output_path, index=False)
        
        summary = f"\n\nProcessed {total} components\nOutput saved to: {output_path}\n\n"
        return summary + "\n".join(results)
        
    except Exception as e:
        return f"Error: {str(e)}"


# --- Gradio Interface Functions ---
def run_agent(query: str, excel_file) -> str:
    """
    Run the agent with user query or process Excel file
    """
    # If Excel file is uploaded, process it
    if excel_file is not None:
        return process_excel_file(excel_file.name)
    
    # Otherwise, run normal query
    try:
        result = agent.run(query)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔧 Component Specification Extraction Agent
    
    **Capabilities:**
    - Extract power rating and heat dissipation from component datasheets
    - Automated web scraping and PDF parsing
    - Batch processing of Excel files
    - General knowledge queries
    
    **Example Queries:**
    - "Extract specs for Conceptpower DPA 120 UL from ABB"
    - "Find power rating for Tesla Model S battery"
    - Upload an Excel file with Component and Company columns for batch processing
    """)
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                lines=3,
                placeholder="Enter your query or upload Excel file below...",
                label="Query Input"
            )
            excel_input = gr.File(
                label="Upload Excel File (Optional)",
                file_types=[".xlsx", ".xls"]
            )
            submit_btn = gr.Button("🚀 Submit", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                lines=20,
                label="Agent Response",
                show_copy_button=True
            )
    
    # Example queries
    gr.Examples(
        examples=[
            ["Extract power rating and heat dissipation for Conceptpower DPA 120 UL from ABB company"],
            ["Search for Tesla Powerwall 2 specifications"],
            ["What is the power rating of iPhone 15 charger?"],
        ],
        inputs=query_input
    )
    
    submit_btn.click(
        fn=run_agent,
        inputs=[query_input, excel_input],
        outputs=output
    )

# --- Launch ---
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Component Specification Extraction Agent - Starting...")
    print("="*80 + "\n")
    demo.launch(share=False)