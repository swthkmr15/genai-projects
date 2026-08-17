from mcp.server.fastmcp import FastMCP
from datetime import datetime
import os

mcp = FastMCP("desktop-assistant")
WORKSPACE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "desktop_workspace")
os.makedirs(WORKSPACE, exist_ok=True)

@mcp.tool()
def get_current_time() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a simple math expression, e.g. '12*7+5'."""
    allowed = set("0123456789+-*/(). %")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) . % are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

@mcp.tool()
def list_files() -> str:
    """List the files saved in the assistant's workspace."""
    files = os.listdir(WORKSPACE)
    return "\n".join(files) if files else "(workspace is empty)"

@mcp.tool()
def write_note(filename: str, content: str) -> str:
    """Save a text note into the workspace."""
    with open(os.path.join(WORKSPACE, filename), "w") as f:
        f.write(content)
    return f"Saved note '{filename}'."

@mcp.tool()
def read_note(filename: str) -> str:
    """Read a text note back from the workspace."""
    path = os.path.join(WORKSPACE, filename)
    if not os.path.exists(path):
        return f"Error: '{filename}' not found."
    with open(path) as f:
        return f.read()

if __name__ == "__main__":
    mcp.run(transport="stdio")