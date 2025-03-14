import os
from pathlib import Path

class Config:
    def __init__(self):
        # Store config file in the project directory instead of user home
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / 'api_key.txt'
        print(f"\nConfig Debug:")
        print(f"Config file path: {self.config_file}")
        print(f"Config file exists: {self.config_file.exists()}")
        self.load_config()
    
    def load_config(self):
        """Load configuration from project file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.api_key = f.read().strip()
                    if not self.api_key:
                        print("Warning: api_key.txt exists but is empty")
                    os.environ['OPENAI_API_KEY'] = self.api_key
            else:
                print(f"Warning: Config file not found at {self.config_file}")
                self.api_key = None
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            self.api_key = None
    
    def save_config(self, api_key):
        """Save API key to project config file"""
        with open(self.config_file, 'w') as f:
            f.write(api_key)
        self.api_key = api_key
        os.environ['OPENAI_API_KEY'] = api_key
