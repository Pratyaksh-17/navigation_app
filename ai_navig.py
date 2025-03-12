import requests

# Hugging Face API for LLM (Free model)
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
HF_HEADERS = {"Authorization": "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"}  # Replace with your Hugging Face API Key

# Function to get real-time location using IP-API
def get_location():
    try:
        response = requests.get("http://ip-api.com/json/").json()
        location_info = f"{response['city']}, {response['regionName']}, {response['country']} (Lat: {response['lat']}, Lon: {response['lon']})"
        return location_info
    except Exception as e:
        return f"Error fetching location: {str(e)}"

# Function to get LLM response
def ask_llm(prompt):
    try:
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json={"inputs": prompt}).json()
        return response.get("generated_text", "No response from AI.")
    except Exception as e:
        return f"Error with AI request: {str(e)}"

# Function to get an address from latitude & longitude (for GPS-based tracking)
def get_address(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        response = requests.get(url).json()
        return response.get("display_name", "Address not found")
    except Exception as e:
        return f"Error fetching address: {str(e)}"

# Main execution
if __name__ == "__main__":
    print("Fetching real-time location...")
    location = get_location()
    print("Your current location:", location)

    print("\nAsking LLM for navigation assistance...")
    prompt = f"The user is currently at {location}. How can you assist them?"
    llm_response = ask_llm(prompt)
    print("\nAI Response:", llm_response)
