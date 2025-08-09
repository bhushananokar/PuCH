import os
from typing import Annotated
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from openai import BaseModel
from pydantic import AnyUrl, Field
import readabilipy
from pathlib import Path

TOKEN = "08a5f9c494a4"
MY_NUMBER = "917588470501"

# Get port from environment variable (Railway sets this)
PORT = int(os.environ.get("PORT", 8085))

class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None

class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(
            public_key=k.public_key, jwks_uri=None, issuer=None, audience=None
        )
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="unknown",
                scopes=[],
                expires_at=None,
            )
        return None

class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        from httpx import AsyncClient, HTTPError

        async with AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except HTTPError as e:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"
                    )
                )
            if response.status_code >= 400:
                raise McpError(
                    ErrorData(
                        code=INTERNAL_ERROR,
                        message=f"Failed to fetch {url} - status code {response.status_code}",
                    )
                )

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = (
            "<html" in page_raw[:100] or "text/html" in content_type or not content_type
        )

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        ret = readabilipy.simple_json.simple_json_from_html_string(
            html, use_readability=True
        )
        if not ret["content"]:
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(
            ret["content"],
            heading_style=markdownify.ATX,
        )
        return content

mcp = FastMCP(
    "My MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, no extra formatting.",
    side_effects=None,
)

@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Return your resume exactly as markdown text.
    
    This function processes your actual resume file and converts it to markdown.
    """
    try:
        # Method 1: Process PDF file
        resume_path = Path("resume.pdf")  # You'll upload this file
        
        if resume_path.exists() and resume_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(resume_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                
                # Convert extracted text to better markdown format
                return format_resume_as_markdown(text)
                
            except ImportError:
                return "Error: PyPDF2 not available for PDF processing"
        
        # Method 2: Process DOCX file
        resume_path = Path("resume.docx")
        if resume_path.exists() and resume_path.suffix.lower() == '.docx':
            try:
                from docx import Document
                doc = Document(resume_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                
                return format_resume_as_markdown(text)
                
            except ImportError:
                return "Error: python-docx not available for DOCX processing"
        
        # Method 3: Process plain text file
        resume_path = Path("resume.txt")
        if resume_path.exists():
            with open(resume_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return format_resume_as_markdown(text)
        
        # Method 4: Process markdown file directly
        resume_path = Path("resume.md")
        if resume_path.exists():
            with open(resume_path, 'r', encoding='utf-8') as file:
                return file.read()
        
        # Fallback: No resume file found
        return """
# Resume Not Found

Please upload your resume as one of these formats:
- resume.pdf
- resume.docx  
- resume.txt
- resume.md

Place the file in the same directory as this server.
"""
        
    except Exception as e:
        return f"Error processing resume: {str(e)}"


def format_resume_as_markdown(raw_text: str) -> str:
    """
    Convert raw extracted text into properly formatted markdown.
    This function intelligently formats the resume text.
    """
    lines = raw_text.strip().split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect and format headers (common resume section titles)
        if any(keyword in line.upper() for keyword in [
            'EXPERIENCE', 'EDUCATION', 'SKILLS', 'PROJECTS', 'SUMMARY', 
            'OBJECTIVE', 'CERTIFICATIONS', 'ACHIEVEMENTS', 'CONTACT'
        ]):
            formatted_lines.append(f"\n## {line}")
        
        # Detect job titles or positions (lines with dates)
        elif any(char in line for char in ['2019', '2020', '2021', '2022', '2023', '2024', '2025']):
            if '-' in line or 'to' in line.lower() or 'present' in line.lower():
                formatted_lines.append(f"\n### {line}")
            else:
                formatted_lines.append(f"- {line}")
        
        # Detect bullet points (lines starting with common indicators)
        elif line.startswith(('•', '-', '*', '◦')) or line.startswith(tuple('0123456789')):
            # Clean up and format as markdown bullet
            clean_line = line.lstrip('•-*◦0123456789. ')
            formatted_lines.append(f"- {clean_line}")
        
        # Regular text lines
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

@mcp.tool
async def validate() -> str:
    """NOTE: This tool must be present in an MCP server used by puch."""
    return MY_NUMBER

FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)

@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[
        int,
        Field(
            default=5000,
            description="Maximum number of characters to return.",
            gt=0,
            lt=1000000,
        ),
    ] = 5000,
    start_index: Annotated[
        int,
        Field(
            default=0,
            description="On return output starting at this character index, useful if a previous fetch was truncated and more context is required.",
            ge=0,
        ),
    ] = 0,
    raw: Annotated[
        bool,
        Field(
            default=False,
            description="Get the actual HTML content if the requested page, without simplification.",
        ),
    ] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_length = len(content)
    if start_index >= original_length:
        content = "<error>No more content available.</error>"
    else:
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            content = "<error>No more content available.</error>"
        else:
            content = truncated_content
            actual_content_length = len(truncated_content)
            remaining_content = original_length - (start_index + actual_content_length)
            if actual_content_length == max_length and remaining_content > 0:
                next_start = start_index + actual_content_length
                content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]

async def main():
    await mcp.run_async(
        "streamable-http",
        host="0.0.0.0",
        port=PORT,  # Use PORT from environment
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
