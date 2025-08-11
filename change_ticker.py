#!/usr/bin/env python3
"""
Script to change ticker symbol in all configuration files
Usage: python change_ticker.py NEW_TICKER
Example: python change_ticker.py SPY
"""

import sys
import os
import re
import json
import glob

def change_ticker(new_ticker):
    """Change ticker in all configuration files"""
    
    new_ticker = new_ticker.upper()
    print(f"🔄 Changing ticker to: {new_ticker}")
    
    # 1. Update src/config.py
    config_file = 'src/config.py'
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Replace TICKER = os.getenv('TICKER', 'ANYTHING')
        content = re.sub(
            r"TICKER = os\.getenv\('TICKER', '[^']+'\)",
            f"TICKER = os.getenv('TICKER', '{new_ticker}')",
            content
        )
        
        # Also replace any standalone TICKER = 'ANYTHING'
        content = re.sub(
            r"TICKER = '[^']+'",
            f"TICKER = '{new_ticker}'",
            content
        )
        
        with open(config_file, 'w') as f:
            f.write(content)
        print(f"✅ Updated {config_file}")
    
    # 2. Update .github/workflows/daily_analysis.yml
    workflow_file = '.github/workflows/daily_analysis.yml'
    if os.path.exists(workflow_file):
        with open(workflow_file, 'r') as f:
            content = f.read()
        
        # Replace default: 'ANYTHING'
        content = re.sub(
            r"default: '[^']+'",
            f"default: '{new_ticker}'",
            content
        )
        
        # Replace || 'ANYTHING'
        content = re.sub(
            r"\|\| '[^']+'",
            f"|| '{new_ticker}'",
            content
        )
        
        with open(workflow_file, 'w') as f:
            f.write(content)
        print(f"✅ Updated {workflow_file}")
    
    # 3. Update .env if it exists
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        with open(env_file, 'w') as f:
            for line in lines:
                if line.startswith('TICKER='):
                    f.write(f'TICKER={new_ticker}\n')
                else:
                    f.write(line)
        print(f"✅ Updated {env_file}")
    
    # 4. Clean data directory
    data_files = glob.glob('data/*')
    if data_files:
        print(f"🗑️  Cleaning {len(data_files)} files in data/")
        for file in data_files:
            try:
                os.remove(file)
                print(f"   Deleted: {file}")
            except:
                pass
    
    # 5. Create a new config file for Streamlit
    streamlit_config = {
        "ticker": new_ticker,
        "changed_at": str(os.path.getmtime('src/config.py'))
    }
    
    with open('ticker_config.json', 'w') as f:
        json.dump(streamlit_config, f)
    print(f"✅ Created ticker_config.json")
    
    print(f"\n✅ Successfully changed ticker to {new_ticker}")
    print("\n📋 Next steps:")
    print("1. Commit and push these changes to GitHub")
    print("2. Go to Streamlit Cloud → Settings → Secrets")
    print(f"3. Change TICKER = \"{new_ticker}\"")
    print("4. Click 'Save' and 'Reboot app'")
    print("5. Clear cache from Streamlit menu")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python change_ticker.py NEW_TICKER")
        print("Example: python change_ticker.py SPY")
        sys.exit(1)
    
    new_ticker = sys.argv[1]
    change_ticker(new_ticker)
