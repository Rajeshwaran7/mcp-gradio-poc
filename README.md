# MCP Gradio Multi-Tool Assistant

A Gradio web interface for interacting with a Model Control Protocol (MCP) server. This application uses Claude 3.5 Sonnet to process user queries and invoke specialized tools through the MCP.

## Features

- **Image Generation**: Generate images from text descriptions using multiple fallback options:
  - Hugging Face Inference API
  - Replicate API
  - SanaSprint Gradio model
  
- **Weather Information**: Get current weather data for any city using OpenWeatherMap API

- **Text Translation**: Translate text between languages (currently using mock implementation)

## Setup

1. Clone this repository
   ```
   git clone <repository-url>
   cd mcp-gradio-poc
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with the following variables:
   ```
   # Required
   ANTHROPIC_API_KEY=your_anthropic_api_key
   
   # Required for weather tool
   OPENWEATHER_API_KEY=your_openweather_api_key
   
   # Optional for image generation (need at least one)
   HF_API_TOKEN=your_huggingface_token
   REPLICATE_API_TOKEN=your_replicate_token
   ```

4. Run the application
   ```
   python app.py
   ```

## Usage

1. Launch the application with `python app.py`
2. Connect to the MCP server by clicking "Connect" (using the default path "gradio_mcp_server.py")
3. Start chatting with the assistant to use the available tools:
   - For images: "Generate an image of mountains at sunset"
   - For weather: "What's the weather in Tokyo?"
   - For translation: "Translate 'hello' to French"

## Project Structure

- `app.py`: Main Gradio interface application
- `gradio_mcp_server.py`: MCP server implementation with tool definitions
- `requirements.txt`: Dependencies
- `.env`: Environment variables and API keys

## Adding New Tools

To add new tools to the MCP server, add a new function with the `@mcp.tool()` decorator in `gradio_mcp_server.py`. The function should:

1. Use async/await syntax
2. Have typed parameters with docstrings
3. Return a JSON string with properly formatted results

Example:
```python
@mcp.tool()
async def my_new_tool(param1: str, param2: int = 42) -> str:
    """Tool description here.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 42)
    """
    result = {}
    # Tool logic here
    return json.dumps(result)
```

## Troubleshooting

- If image generation fails, make sure you have at least one of the API keys set up
- For weather data, an OpenWeatherMap API key is required
- For any API errors, check the console output for detailed error messages 