import os
import json
import asyncio
from typing import List, Dict, Any, Union
import ssl
import socket
import shutil
from datetime import datetime
from pathlib import Path

import gradio as gr
from gradio.components.chatbot import ChatMessage
from anthropic import Anthropic
from dotenv import load_dotenv
from gradio_client import Client

# Create a folder for saving generated images if it doesn't exist
IMAGES_FOLDER = os.path.join(os.getcwd(), "generated_images")
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Import the generate_image function directly from gradio_mcp_server.py
# But we'll need to create a wrapper to handle SSL issues
from gradio_mcp_server import generate_image as original_generate_image

# Create a wrapper for generate_image with better error handling
async def generate_image(prompt: str, width: int = 512, height: int = 512) -> str:
    """Generate an image using the Hugging Face space API directly and save to project folder"""
    try:
        print(f"Generating image with prompt: '{prompt}', width: {width}, height: {height}")
        
        # Set socket timeout globally instead of Client parameter
        original_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(120)  # 120 second timeout
        
        try:
            # Use basic Client initialization without extra parameters
            client = Client("https://ysharma-sanasprint.hf.space/")
            
            print("Calling Hugging Face API...")
            # The API expects these specific parameters in this order
            result = client.predict(
                prompt,              # text prompt
                "0.6B",              # model size
                0,                   # seed (0 = random)
                True,                # use_karras_scheduler
                width,               # width
                height,              # height
                4.0,                 # guidance_scale
                2,                   # num_inference_steps
                api_name="/infer"
            )
            
            print(f"Received response type: {type(result)}")
            print(f"Response content: {result}")
            
            # Based on the SanaSprint space documentation, the result should be 
            # a list or tuple containing image data. The expected format is:
            # result = [{'url': 'https://path-to-image.jpg'}, ...other metadata...]
            # or result = ({'url': 'https://path-to-image.jpg'}, ...other metadata...)
            
            # Convert tuple to list if needed
            if isinstance(result, tuple):
                print("Converting tuple response to list")
                result_list = list(result)
                
                # Special case for local file paths (common in newer gradio client versions)
                if len(result) >= 1 and isinstance(result[0], str) and (
                    result[0].endswith('.png') or result[0].endswith('.jpg') or 
                    result[0].endswith('.webp') or '\\gradio\\' in result[0]):
                    
                    temp_image_path = result[0]
                    print(f"Found local file path in result: {temp_image_path}")
                    
                    # Generate a new filename in our project folder
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_filename = f"image_{timestamp}.webp"
                    new_image_path = os.path.join(IMAGES_FOLDER, image_filename)
                    
                    # Copy the image from temp location to our project folder
                    try:
                        shutil.copy2(temp_image_path, new_image_path)
                        print(f"Copied image to: {new_image_path}")
                        
                        # Use this new path in the response
                        response = {
                            "type": "image",
                            "url": new_image_path,
                            "message": f"Generated image for prompt: {prompt}"
                        }
                        return json.dumps(response)
                    except Exception as copy_err:
                        print(f"Error copying image: {str(copy_err)}")
                        # Fall back to using the original temp path
                        response = {
                            "type": "image",
                            "url": temp_image_path,
                            "message": f"Generated image for prompt: {prompt}"
                        }
                        return json.dumps(response)
            else:
                result_list = result
            
            if result_list and (isinstance(result_list, list) or isinstance(result_list, tuple)):
                # Try to extract image URL based on known format from SanaSprint
                image_data = None
                
                # First, check if it's in the expected format with a dict containing 'url'
                if len(result_list) > 0 and isinstance(result_list[0], dict) and 'url' in result_list[0]:
                    image_data = result_list[0]
                    print(f"Found image data in expected format: {image_data}")
                
                # If we have image data with a URL, return it
                if image_data and 'url' in image_data:
                    response = {
                        "type": "image",
                        "url": image_data['url'],
                        "message": f"Generated image for prompt: {prompt}"
                    }
                    print(f"Successfully extracted image URL: {image_data['url']}")
                    return json.dumps(response)
                
                # If the first element is directly a URL string
                elif len(result_list) > 0 and isinstance(result_list[0], str) and (
                    result_list[0].startswith('http://') or result_list[0].startswith('https://')
                ):
                    response = {
                        "type": "image",
                        "url": result_list[0],
                        "message": f"Generated image for prompt: {prompt}"
                    }
                    print(f"Extracted image URL from first element: {result_list[0]}")
                    return json.dumps(response)
                    
                # Special case where image might be in a nested format
                elif len(result_list) > 0 and (isinstance(result_list[0], list) or isinstance(result_list[0], tuple)) and len(result_list[0]) > 0:
                    # Convert to list if tuple
                    first_item = list(result_list[0]) if isinstance(result_list[0], tuple) else result_list[0]
                    
                    # Check first item of nested list
                    if len(first_item) > 0 and isinstance(first_item[0], str) and (
                        first_item[0].startswith('http://') or first_item[0].startswith('https://')
                    ):
                        response = {
                            "type": "image",
                            "url": first_item[0],
                            "message": f"Generated image for prompt: {prompt}"
                        }
                        print(f"Extracted image URL from nested list: {first_item[0]}")
                        return json.dumps(response)
                        
                    # Check if it's a dict in a nested list
                    elif len(first_item) > 0 and isinstance(first_item[0], dict) and 'url' in first_item[0]:
                        response = {
                            "type": "image",
                            "url": first_item[0]['url'],
                            "message": f"Generated image for prompt: {prompt}"
                        }
                        print(f"Extracted image URL from nested dict: {first_item[0]['url']}")
                        return json.dumps(response)
                
                # Print detailed structure for debugging
                print("Could not find image URL in expected locations. Detailed structure:")
                import pprint
                pprint.pprint(result)
                
                # Return the entire result for debugging
                return json.dumps({
                    "type": "error",
                    "message": f"Could not extract image URL from response. Raw response (first 500 chars): {str(result)[:500]}"
                })
            else:
                print(f"Unexpected response format. Expected list or tuple, got: {type(result)}")
                return json.dumps({
                    "type": "error",
                    "message": f"Unexpected response format from API. Expected list or tuple, got: {type(result)}"
                })
            
        except Exception as e:
            print(f"Error in image generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return json.dumps({
                "type": "error",
                "message": f"Error generating image: {str(e)}"
            })
    finally:
        # Always restore original timeout
        socket.setdefaulttimeout(original_timeout)

load_dotenv()

class MCPClientWrapper:
    def __init__(self):
        self.anthropic = Anthropic()
        # Pre-defined tools for different services
        self.tools_by_service = {
            "weather": [
                {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            ],
            "image": [
                # No tools for image generation service since we're handling it directly
            ],
            "general": [
                {
                    "name": "get_weather",
                    "description": "Get weather information for a location",
                    "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
                # No image generation tool here either
            ]
        }
        self.active_service = "general"
    
    def set_active_service(self, service_name: str, return_message=True):
        """Set the active service for tool selection"""
        if service_name in self.tools_by_service:
            self.active_service = service_name
            if return_message:
                return f"Active service set to: {service_name}"
            return None
        if return_message:
            return f"Unknown service: {service_name}"
        return None
    
    async def process_message(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]], service_name: str = None) -> tuple:
        """Async message processing for Gradio to call directly"""
        # Update active service if specified
        if service_name and service_name in self.tools_by_service:
            self.active_service = service_name
            
        # Handle image generation directly without Claude
        if service_name == "image" or (self.active_service == "image" and not service_name):
            result_messages = await self._process_image_generation(message, history)
        else:
            result_messages = await self._process_query(message, history)
            
        return history + [{"role": "user", "content": message}] + result_messages, gr.Textbox(value="")
    
    async def _process_image_generation(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]):
        """Process image generation requests directly without using Claude"""
        result_messages = []
        
        # First message explains what we're doing
        result_messages.append({
            "role": "assistant",
            "content": f"I'll generate an image based on your description: '{message}'",
            "metadata": {
                "title": "Using image generation",
                "log": f"Parameters: {{\"prompt\": \"{message}\"}}",
                "status": "pending",
                "id": "tool_call_generate_image"
            }
        })
        
        try:
            # Call the image generation function directly
            result_content = await generate_image(message)
            print(result_content,"result_content")
            
            result_messages[0]["metadata"]["status"] = "done"
            
            result_messages.append({
                "role": "assistant",
                "content": "Here is the generated image:",
                "metadata": {
                    "title": "Image Generation Result",
                    "status": "done",
                    "id": "result_generate_image"
                }
            })
            
            try:
                result_json = json.loads(result_content)
                if isinstance(result_json, dict) and "type" in result_json:
                    # Handle image generation results
                    if result_json["type"] == "image" and "url" in result_json:
                        image_url = result_json["url"]
                        alt_text = result_json.get("message", "Generated image")
                        
                        # Display the image in the chat
                        result_messages.append({
                            "role": "assistant",
                            "content": {"path": image_url, "alt_text": alt_text},
                            "metadata": {
                                "parent_id": "result_generate_image",
                                "id": "image_generate_image",
                                "title": "Generated Image"
                            }
                        })
                        
                        # Add a message showing where the image is saved for reference
                        if os.path.exists(image_url) and IMAGES_FOLDER in image_url:
                            relative_path = os.path.relpath(image_url, os.getcwd())
                            result_messages.append({
                                "role": "assistant",
                                "content": f"✅ Image saved to project folder: {relative_path}",
                                "metadata": {
                                    "parent_id": "result_generate_image",
                                    "id": "image_path_info",
                                    "title": "Image File Location"
                                }
                            })
                            
                            # Add an instruction to open the file
                            result_messages.append({
                                "role": "assistant",
                                "content": "You can find the image in the 'generated_images' folder of your project.",
                                "metadata": {
                                    "parent_id": "result_generate_image",
                                    "id": "open_instructions",
                                    "title": "Access Instructions"
                                }
                            })
                    elif result_json["type"] == "error":
                        error_message = result_json.get("message", "Unknown error occurred")
                        result_messages.append({
                            "role": "assistant",
                            "content": f"Error: {error_message}",
                            "metadata": {
                                "parent_id": "result_generate_image",
                                "id": "error_generate_image",
                                "title": "Error",
                                "status": "done"
                            }
                        })
                else:
                    result_messages.append({
                        "role": "assistant",
                        "content": "```\n" + result_content + "\n```",
                        "metadata": {
                            "parent_id": "result_generate_image",
                            "id": "raw_result_generate_image",
                            "title": "Raw Output"
                        }
                    })
            except Exception as parse_err:
                print(f"Error parsing result: {str(parse_err)}")
                result_messages.append({
                    "role": "assistant",
                    "content": "```\n" + result_content + "\n```",
                    "metadata": {
                        "parent_id": "result_generate_image",
                        "id": "raw_result_generate_image",
                        "title": "Raw Output"
                    }
                })
                
        except Exception as e:
            result_messages[0]["metadata"]["status"] = "done"
            
            error_message = f"Error generating image: {str(e)}"
            result_messages.append({
                "role": "assistant",
                "content": error_message,
                "metadata": {
                    "title": "Image Generation Error",
                    "status": "done",
                    "id": "error_generate_image"
                }
            })
        
        return result_messages
    
    async def _process_query(self, message: str, history: List[Union[Dict[str, Any], ChatMessage]]):
        claude_messages = []
        for msg in history:
            if isinstance(msg, ChatMessage):
                role, content = msg.role, msg.content
            else:
                role, content = msg.get("role"), msg.get("content")
            
            if role in ["user", "assistant", "system"]:
                claude_messages.append({"role": role, "content": content})
        
        claude_messages.append({"role": "user", "content": message})
        
        # Use tools based on the active service
        active_tools = self.tools_by_service.get(self.active_service, [])
        
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=claude_messages,
            tools=active_tools
        )

        result_messages = []
        
        for content in response.content:
            if content.type == 'text':
                result_messages.append({
                    "role": "assistant", 
                    "content": content.text
                })
                
            elif content.type == 'tool_use':
                tool_name = content.name
                tool_args = content.input
                
                result_messages.append({
                    "role": "assistant",
                    "content": f"I'll use the {tool_name} tool to help answer your question.",
                    "metadata": {
                        "title": f"Using tool: {tool_name}",
                        "log": f"Parameters: {json.dumps(tool_args, ensure_ascii=True)}",
                        "status": "pending",
                        "id": f"tool_call_{tool_name}"
                    }
                })
                
                # Simulate tool calls with mock responses
                try:
                    if tool_name == "get_weather":
                        location = tool_args.get("location", "Unknown")
                        # Mock weather data
                        weather_data = {
                            "type": "weather",
                            "data": {
                                "location": location,
                                "temperature": 22,
                                "condition": "Partly Cloudy",
                                "humidity": 65
                            }
                        }
                        result_content = json.dumps(weather_data)
                    else:
                        # Generic mock response
                        result_content = json.dumps({"type": "text", "content": "Mock response for " + tool_name})
                    
                    if result_messages and "metadata" in result_messages[-1]:
                        result_messages[-1]["metadata"]["status"] = "done"
                    
                    result_messages.append({
                        "role": "assistant",
                        "content": "Here are the results from the tool:",
                        "metadata": {
                            "title": f"Tool Result for {tool_name}",
                            "status": "done",
                            "id": f"result_{tool_name}"
                        }
                    })
                    
                    try:
                        result_json = json.loads(result_content)
                        if isinstance(result_json, dict) and "type" in result_json:
                            # Handle weather results
                            if result_json["type"] == "weather" and "data" in result_json:
                                weather_data = result_json["data"]
                                formatted_weather = f"**Weather Report**\n\nLocation: {weather_data.get('location', 'Unknown')}\nTemperature: {weather_data.get('temperature', 'N/A')}°C\nCondition: {weather_data.get('condition', 'N/A')}\nHumidity: {weather_data.get('humidity', 'N/A')}%"
                                
                                result_messages.append({
                                    "role": "assistant",
                                    "content": formatted_weather,
                                    "metadata": {
                                        "parent_id": f"result_{tool_name}",
                                        "id": f"weather_{tool_name}",
                                        "title": "Weather Information"
                                    }
                                    })
                            else:
                                result_messages.append({
                                    "role": "assistant",
                                    "content": "```\n" + result_content + "\n```",
                                    "metadata": {
                                        "parent_id": f"result_{tool_name}",
                                        "id": f"raw_result_{tool_name}",
                                        "title": "Raw Output"
                                    }
                                })
                        else:
                            result_messages.append({
                                "role": "assistant",
                                "content": "```\n" + result_content + "\n```",
                                "metadata": {
                                    "parent_id": f"result_{tool_name}",
                                    "id": f"raw_result_{tool_name}",
                                    "title": "Raw Output"
                                }
                            })
                    except:
                        result_messages.append({
                            "role": "assistant",
                            "content": "```\n" + result_content + "\n```",
                            "metadata": {
                                "parent_id": f"result_{tool_name}",
                                "id": f"raw_result_{tool_name}",
                                "title": "Raw Output"
                            }
                        })
                    
                    claude_messages.append({"role": "user", "content": f"Tool result for {tool_name}: {result_content}"})
                    
                except Exception as e:
                    if result_messages and "metadata" in result_messages[-1]:
                        result_messages[-1]["metadata"]["status"] = "done"
                    
                    error_message = f"Error while calling {tool_name} tool: {str(e)}"
                    result_messages.append({
                        "role": "assistant",
                        "content": error_message,
                        "metadata": {
                            "title": "Tool Error",
                            "status": "done",
                            "id": f"error_{tool_name}"
                        }
                    })
                    
                    claude_messages.append({"role": "user", "content": f"Tool error for {tool_name}: {error_message}"})
                
                # Get Claude's response about the result or error
                next_response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=claude_messages,
                )
                
                if next_response.content and next_response.content[0].type == 'text':
                    result_messages.append({
                        "role": "assistant",
                        "content": next_response.content[0].text
                    })

        return result_messages

client = MCPClientWrapper()

# Predefined service configurations
SERVICE_CONFIGS = {
    "weather": {
        "description": "Get weather information for any location",
        "placeholder": "Ask about the weather (e.g., What's the weather in Paris?)",
        "example": "What's the current weather in Tokyo?"
    },
    "image": {
        "description": "Generate images using AI models",
        "placeholder": "Describe the image you want to generate",
        "example": "Generate an image of a sunset over mountains"
    },
    "general": {
        "description": "Access multiple tools including weather and image generation",
        "placeholder": "Ask any question using available tools",
        "example": "What's the weather in London? Also, generate an image of rainy London streets."
    }
}

def gradio_interface():
    with gr.Blocks(title="Multi-Tool Assistant") as demo:
        gr.Markdown("# Multi-Tool Assistant")
        gr.Markdown("Access various AI tools including weather information and image generation")
        
        with gr.Tabs() as tabs:
            # Weather tab
            with gr.Tab("Weather Assistant"):
                weather_info = gr.Markdown(f"## Weather Information\n{SERVICE_CONFIGS['weather']['description']}")
                
                weather_chatbot = gr.Chatbot(
                    value=[], 
                    height=500,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=("👤", "🌦️")
                )
                
                weather_examples = gr.Examples(
                    examples=[[SERVICE_CONFIGS['weather']['example']]],
                    inputs=[weather_msg := gr.Textbox(
                        label="Your Weather Question",
                        placeholder=SERVICE_CONFIGS['weather']['placeholder'],
                        scale=4
                    )]
                )
                
                weather_clear_btn = gr.Button("Clear Chat", scale=1)
                
                # Set weather service active on tab selection
                tabs.select(lambda: client.set_active_service("weather", return_message=False), None, None, js="(i) => i === 0")
                
                weather_msg.submit(
                    client.process_message, 
                    [weather_msg, weather_chatbot, gr.State("weather")], 
                    [weather_chatbot, weather_msg]
                )
                weather_clear_btn.click(lambda: [], None, weather_chatbot)
                
            # Image generation tab
            with gr.Tab("Image Generator"):
                image_info = gr.Markdown(f"## Image Generation\n{SERVICE_CONFIGS['image']['description']}")
                
                with gr.Row():
                    # Chat interface on the left
                    with gr.Column(scale=2):
                        image_chatbot = gr.Chatbot(
                            value=[], 
                            height=500,
                            type="messages",
                            show_copy_button=True,
                            avatar_images=("👤", "🖼️")
                        )
                        
                        image_examples = gr.Examples(
                            examples=[[SERVICE_CONFIGS['image']['example']]],
                            inputs=[image_msg := gr.Textbox(
                                label="Your Image Request",
                                placeholder=SERVICE_CONFIGS['image']['placeholder'],
                                scale=4
                            )]
                        )
                        
                        image_clear_btn = gr.Button("Clear Chat", scale=1)
                    
                    # Image gallery on the right
            with gr.Column(scale=1):
                        gr.Markdown("### Generated Images")
                        image_gallery = gr.Gallery(
                            label="Recent Images",
                            object_fit="contain",
                            columns=2,
                            height=400
                        )
                        refresh_gallery = gr.Button("Refresh Gallery")
                
                def update_image_gallery():
                    image_files = []
                    if os.path.exists(IMAGES_FOLDER):
                        for file in os.listdir(IMAGES_FOLDER):
                            if file.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                                image_files.append(os.path.join(IMAGES_FOLDER, file))
                    return image_files
                
                # Set image service active on tab selection
                tabs.select(lambda: client.set_active_service("image", return_message=False), None, None, js="(i) => i === 1")
                
                # Update gallery when tab is selected
                tabs.select(update_image_gallery, None, image_gallery, js="(i) => i === 1")
                
                # Connect the refresh button to update the gallery
                refresh_gallery.click(update_image_gallery, None, image_gallery)
                
                image_msg.submit(
                    client.process_message, 
                    [image_msg, image_chatbot, gr.State("image")], 
                    [image_chatbot, image_msg]
                )
                
                # Also update gallery after generating a new image
                image_msg.submit(
                    update_image_gallery,
                    None,
                    image_gallery
                )
                
                image_clear_btn.click(lambda: [], None, image_chatbot)
                
            # Combined services tab
            with gr.Tab("All Services"):
                combined_info = gr.Markdown(f"## All Services\n{SERVICE_CONFIGS['general']['description']}")
                
                combined_chatbot = gr.Chatbot(
            value=[], 
            height=500,
            type="messages",
            show_copy_button=True,
            avatar_images=("👤", "🤖")
        )
        
                combined_examples = gr.Examples(
                    examples=[[SERVICE_CONFIGS['general']['example']]],
                    inputs=[combined_msg := gr.Textbox(
                label="Your Question",
                        placeholder=SERVICE_CONFIGS['general']['placeholder'],
                scale=4
                    )]
                )
                
                combined_clear_btn = gr.Button("Clear Chat", scale=1)
                
                # Set general service active on tab selection
                tabs.select(lambda: client.set_active_service("general", return_message=False), None, None, js="(i) => i === 2")
                
                combined_msg.submit(
                    client.process_message, 
                    [combined_msg, combined_chatbot, gr.State("general")], 
                    [combined_chatbot, combined_msg]
                )
                combined_clear_btn.click(lambda: [], None, combined_chatbot)
        
    return demo

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not found in environment. Please set it in your .env file.")
    
    interface = gradio_interface()
    interface.launch(debug=True, share=True)