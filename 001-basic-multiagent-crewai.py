import requests
import datetime
import sys
from contextlib import redirect_stdout
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

# llm = ChatOpenAI(model="gpt-4-turbo-preview")

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-groq-70b-8192-tool-use-preview"
)

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for file in self.files:
            file.write(obj)
            file.flush()  # Opcional: asegura que se escriba inmediatamente

    def flush(self):
        for file in self.files:
            file.flush()

@tool("process_search_tool", return_direct=False)
def process_search_tool(url: str) -> str:
    """Used to process content found on the internet."""
    response = requests.get(url=url)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text()

tools = [TavilySearchResults(max_results=1), process_search_tool]

online_researcher = Agent(
    role="Online Researcher",
    goal="Conduct focused and efficient online research",
    backstory="""You are a precise and efficient research analyst who:
    1. Focuses on finding the most recent and authoritative sources
    2. Prioritizes official announcements and peer-reviewed research
    3. Extracts only the most relevant information, avoiding redundancy
    4. Validates information across multiple sources before reporting
    5. Presents findings in clear, structured bullet points
    
    You never speculate or include unverified information. You limit your research to 3-4 
    key sources and present only confirmed, significant developments.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

blog_manager = Agent(
    role="Blog Manager",
    goal="Create concise, impactful blog content",
    backstory="""You are a focused content strategist who:
    1. Transforms complex research into clear, accessible content
    2. Structures articles with a clear beginning, middle, and end
    3. Prioritizes actionable insights over general information
    4. Uses simple language to explain complex topics
    5. Maintains strict article length limits
    
    You consistently produce articles that:
    - Start with a strong hook
    - Present 2-3 key points maximum
    - End with clear takeaways
    - Stay under 500 words total
    - Include only the most relevant source links""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

social_media_manager = Agent(
    role="Social Media Manager",
    goal="Create viral-worthy tweets",
    backstory="""You are a social media expert who:
    1. Identifies the single most compelling angle from content
    2. Creates memorable, shareable messages
    3. Uses maximum 2 relevant hashtags
    4. Focuses on one key message per tweet
    5. Optimizes engagement through question or call-to-action
    
    You never exceed 280 characters and always prioritize clarity 
    over cleverness. Each tweet should stand alone without needing 
    external context.""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

content_marketing_manager = Agent(
    role="Content Marketing Manager",
    goal="Ensure quality and consistency across all content",
    backstory="""You are a detail-oriented content supervisor who:
    1. Ensures factual accuracy and clarity
    2. Maintains consistent tone across all formats
    3. Verifies source attribution
    4. Checks for conciseness and impact
    5. Guarantees content alignment with objectives
    
    You focus on:
    - Removing any redundant information
    - Ensuring each piece serves its specific purpose
    - Maintaining professional standards
    - Verifying all claims have proper support""",
    verbose=True,
    allow_delegation=True,
    tools=tools,
    llm=llm
)

spanish_translator = Agent(
    role="Spanish Translator",
    goal="Provide accurate Spanish (Spain) translations while maintaining tone and context",
    backstory="""You are an expert translator specialized in Spanish (Spain) localization who:
    1. Maintains the original meaning and tone
    2. Uses Spain-specific terminology and expressions
    3. Adapts cultural references appropriately
    4. Preserves formatting and structure
    5. Ensures natural-sounding Spanish text
    
    You focus on:
    - Using proper Spanish (Spain) idioms
    - Maintaining technical accuracy
    - Preserving source formatting
    - Adapting hashtags appropriately""",
    verbose=True,
    allow_delegation=False,  # El traductor no necesita delegar
    tools=tools,
    llm=llm
)

task1 = Task(
    description="""Research 3 key AI developments (last 3 months only):
    1. One major technical breakthrough
    2. One proven business application
    3. One societal impact
    
    Rules:
    - Use only top-tier sources (e.g., arXiv, company releases)
    - 100 words max per finding
    - Include source links
    - Focus on verified results only""",
    expected_output="3-point AI report with sources",
    agent=online_researcher
)

task2 = Task(
    description="""Create a 3-paragraph blog post:
    P1: Highlight top breakthrough (50 words)
    P2: Explain real-world impact (100 words)
    P3: Future implications (50 words)
    
    Rules:
    - Start with the most impactful finding
    - Include max 3 source links
    - Use clear, non-technical language
    - Focus on practical applications""",
    expected_output="300-word blog post",
    agent=blog_manager
)

task3 = Task(
    description="""Craft one tweet with:
    - The single most important finding
    - Two relevant #hashtags
    - One clear call-to-action
    
    Rules:
    - Max 280 characters
    - Focus on business impact
    - Make it shareable""",
    expected_output="High-impact tweet",
    agent=social_media_manager
)

# Modificar task4 para incluir todos los outputs
task4 = Task(
    description="""Review and compile all outputs:
    1. Include the original research findings (labeled as "RESEARCH FINDINGS:")
    2. Include the blog post (labeled as "BLOG POST:")
    3. Include the tweet (labeled as "TWEET:")
    4. Add your quality verification (labeled as "QUALITY VERIFICATION:")
    
    Present everything in this order with clear separators and labels.""",
    expected_output="Complete report with all outputs",
    agent=content_marketing_manager
)

# Añadir la tarea de traducción
task5 = Task(
    description="""Translate the complete report to Spanish (Spain):
    
    Rules:
    - Maintain all section labels in Spanish (HALLAZGOS DE LA INVESTIGACIÓN:, ENTRADA DE BLOG:, TWEET:, VERIFICACIÓN DE CALIDAD:)
    - Adapt hashtags appropriately
    - Keep any URLs or technical terms as needed
    - Use Spain-specific terminology
    - Maintain the original formatting
    
    Present the translation with the same structure and clarity as the original.""",
    expected_output="Complete report in Spanish (Spain)",
    agent=spanish_translator
)

agents = [online_researcher, blog_manager, social_media_manager, content_marketing_manager, spanish_translator]
tasks = [task1, task2, task3, task4, task5]

crew = Crew(
    agents=agents,
    tasks=tasks,
    verbose=2
)

def get_log_filename():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"crewai_log_{timestamp}.txt"

log_filename = get_log_filename()
print(f"Guardando log en: {log_filename}")

with open(log_filename, 'w', encoding='utf-8') as log_file:
    # Guardar la referencia original de stdout
    original_stdout = sys.stdout
    # Crear un Tee que escriba tanto a stdout como al archivo
    sys.stdout = Tee(sys.stdout, log_file)
    
    try:
        result = crew.kickoff()
        print("\n=== RESULTADO FINAL ===")
        print(result)
    except Exception as e:
        print("\n=== ERROR ===")
        print(f"Error durante la ejecución: {str(e)}")
    finally:
        # Restaurar stdout original
        sys.stdout = original_stdout

print(f"\nLog guardado en: {log_filename}")