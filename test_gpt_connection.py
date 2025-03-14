import os
from openai import OpenAI
from config import Config

def test_connection():
    print("Testing OpenAI connection...")
    
    try:
        # Load API key
        config = Config()
        
        # Debug: Check if key exists and its format
        print("\nDebug Info:")
        print(f"API key exists: {config.api_key is not None}")
        print(f"API key starts with: {config.api_key[:7]}..." if config.api_key else "No key found")
        print(f"API key length: {len(config.api_key)}" if config.api_key else "No key")
        print(f"Environment variable set: {os.getenv('OPENAI_API_KEY') is not None}")
        
        if not config.api_key:
            raise ValueError("API key not found in api_key.txt")
            
        print("\n✓ API key loaded successfully")
        
        # Initialize OpenAI client
        print("\nInitializing OpenAI client...")
        client = OpenAI(api_key=config.api_key)
        
        # Test basic API connection
        print("Sending test request to OpenAI API...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Respond with exactly: 'Connection successful'"}
            ]
        )
        
        print("\nResponse from GPT:")
        print(response.choices[0].message.content)
        
        if "Connection successful" in response.choices[0].message.content:
            print("\n✓ Successfully verified OpenAI API connection")
        else:
            print("\n! Response received but unexpected content")
        
    except ValueError as ve:
        print(f"\n❌ Configuration Error: {str(ve)}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False
        
    return True

if __name__ == "__main__":
    test_connection() 