from mcp.server.fastmcp import FastMCP
import json
import sys
import io
import time
from gradio_client import Client

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

mcp = FastMCP("huggingface_spaces_image_display")

@mcp.tool()
async def generate_image(prompt: str, width: int = 512, height: int = 512) -> str:
    """Generate an image using SanaSprint model.
    
    Args:
        prompt: Text prompt describing the image to generate
        width: Image width (default: 512)
        height: Image height (default: 512)
    """
    client = Client("https://ysharma-sanasprint.hf.space/")
    
    try:
        result = client.predict(
            prompt,
            "0.6B",
            0,
            True,
            width,
            height,
            4.0,
            2,
            api_name="/infer"
        )
        
        if isinstance(result, list) and len(result) >= 1:
            image_data = result[0]
            if isinstance(image_data, dict) and "url" in image_data:
                return json.dumps({
                    "type": "image",
                    "url": image_data["url"],
                    "message": f"Generated image for prompt: {prompt}"
                })
        
        return json.dumps({
            "type": "error",
            "message": "Failed to generate image"
        })
        
    except Exception as e:
        return json.dumps({
            "type": "error",
            "message": f"Error generating image: {str(e)}"
        })

if __name__ == "__main__":
    mcp.run(transport='stdio')