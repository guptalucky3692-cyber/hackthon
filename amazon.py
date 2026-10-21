import os
import json
import boto3
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="VoguePulse AI - Alexa+ MCP Server")

# Initialize AWS Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# Sample digital wardrobe dataset
WARDROBE_DATA = [ {"id": 1, "item": "Navy Blue Tailored Blazer", "category": "outerwear", "style": "formal"}, {"id": 2, "item": "Crisp White Oxford Shirt", "category": "top", "style": "formal/smart-casual"},{"id": 3, "item": "Charcoal Grey Slim Trousers", "category": "bottom", "style": "formal"},
    {"id": 4, "item": "Beige Knit Crewneck Sweater", "category": "top", "style": "casual"},
    {"id": 5, "item": "Dark Wash Denim Jeans", "category": "bottom", "style": "casual"},
    {"id": 6, "item": "Classic Brown Leather Loafers", "category": "footwear", "style": "smart-casual"}
]

@app.get("/")
def root():
    return {"status": "online", "service": "VoguePulse AI MCP Server", "spec": "2025-11-25"}

@app.post("/mcp/tools/list")
def list_mcp_tools():
    """Declares available tools to Alexa+ via MCP spec."""
    return {
        "tools": [
            {
                "name": "list_wardrobe",
                "description": "Fetches items from the user's registered wardrobe.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "description": "top, bottom, outerwear, footwear"}
                    }
                }
            },
            {
                "name": "get_outfit_recommendation",
                "description": "Generates a cohesive outfit recommendation based on occasion and weather.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "occasion": {"type": "string", "description": "e.g., business casual, dinner party"},
                        "weather": {"type": "string", "description": "e.g., chilly, warm, rainy"}
                    },
                    "required": ["occasion"]
                }
            }
        ]
    }

@app.post("/mcp/tools/call")
async def call_mcp_tool(request: Request):
    """Executes the specific tool call dispatched by the Alexa+ agent."""
    payload = await request.json()
    tool_name = payload.get("name")
    args = payload.get("arguments", {})

    if tool_name == "list_wardrobe":
        category = args.get("category")
        if category:
            results = [item for item in WARDROBE_DATA if item["category"] == category]
        else:
            results = WARDROBE_DATA
        return {"content": [{"type": "text", "text": json.dumps(results)}]}

    elif tool_name == "get_outfit_recommendation":
        occasion = args.get("occasion", "casual")
        weather = args.get("weather", "temperate")

        prompt = f"""
        User Wardrobe Inventory:
        {json.dumps(WARDROBE_DATA)}

        Occasion: {occasion}
        Weather: {weather}

        Task: Select a coordinated outfit from the inventory suitable for this occasion and weather. Provide a short, direct explanation.
        """

        try:
            bedrock_payload = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}]
            })
            response = bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=bedrock_payload
            )
            response_body = json.loads(response["body"].read())
            advice = response_body["content"][0]["text"]
        except Exception:
            advice = f"Pair the Navy Blue Tailored Blazer with the Crisp White Oxford Shirt and Charcoal Grey Slim Trousers for a sharp, confident {occasion} look."

        return {"content": [{"type": "text", "text": advice}]}

    return JSONResponse(status_code=404, content={"error": f"Tool '{tool_name}' not recognized."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)