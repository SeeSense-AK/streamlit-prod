#!/usr/bin/env python3
"""
Deployment script for SeeSense Dashboard
Handles setup, validation, and deployment tasks
"""
import subprocess
import sys
from pathlib import Path
import argparse
import shutil

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    try:
        import streamlit
        import pandas
        import plotly
        import folium
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("📦 Install requirements with: pip install -r requirements.txt")
        return False

def create_directory_structure():
    """Create necessary directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "logs",
        "assets/styles"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep files to preserve empty directories
        gitkeep_file = Path(directory) / ".gitkeep"
        if not gitkeep_file.exists():
            gitkeep_file.touch()
    
    print("✅ Directory structure created")

def check_data_files():
    """Check if data files are present"""
    print("📊 Checking for data files...")
    
    required_files = [
        "data/raw/routes.csv",
        "data/raw/braking_hotspots.csv", 
        "data/raw/swerving_hotspots.csv",
        "data/raw/time_series.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing {len(missing_files)} data files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        print("\n💡 Options to get data:")
        print("   1. Add your own CSV files to data/raw/")
        print("   2. Generate sample data: python scripts/generate_sample_data.py")
        return False
    else:
        print("✅ All data files present")
        return True

def generate_sample_data():
    """Generate sample data if requested"""
    print("🎲 Generating sample data...")
    
    try:
        subprocess.run([
            sys.executable, "scripts/generate_sample_data.py", 
            "--location", "london",
            "--routes", "500",
            "--days", "180"
        ], check=True)
        print("✅ Sample data generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate sample data: {e}")
        return False

def validate_installation():
    """Validate the installation"""
    print("🔍 Validating installation...")
    
    # Check if main app file exists
    if not Path("app/main.py").exists():
        print("❌ Main application file not found: app/main.py")
        return False
    
    # Check config files
    if not Path("config/settings.yaml").exists():
        print("❌ Configuration file not found: config/settings.yaml")
        return False
    
    print("✅ Installation validation passed")
    return True

def run_dashboard(port=8501, host="localhost"):
    """Run the dashboard"""
    print(f"🚀 Starting dashboard on {host}:{port}")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app/main.py",
            f"--server.port={port}",
            f"--server.address={host}",
            "--server.headless=true"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description="Deploy SeeSense Dashboard")
    parser.add_argument("--setup-only", action="store_true",
                       help="Only setup, don't run dashboard")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate sample data if missing")
    parser.add_argument("--port", type=int, default=8501,
                       help="Port to run dashboard on")
    parser.add_argument("--host", default="localhost",
                       help="Host to run dashboard on")
    parser.add_argument("--production", action="store_true",
                       help="Production deployment mode")
    
    args = parser.parse_args()
    
    print("🚲 SeeSense Dashboard Deployment")
    print("=" * 40)
    
    # Step 1: Check requirements
    if not check_requirements():
        return 1
    
    # Step 2: Create directory structure
    create_directory_structure()
    
    # Step 3: Validate installation
    if not validate_installation():
        return 1
    
    # Step 4: Check data files
    data_present = check_data_files()
    
    if not data_present and args.generate_data:
        if not generate_sample_data():
            return 1
    elif not data_present:
        print("\n⚠️  No data files found!")
        print("🔧 Run with --generate-data to create sample data")
        print("📁 Or add your CSV files to data/raw/")
        
        if not args.setup_only:
            response = input("\nContinue without data? Dashboard will show setup page. (y/N): ")
            if response.lower() != 'y':
                return 1
    
    # Step 5: Setup complete
    print("\n✅ Setup complete!")
    
    if args.setup_only:
        print("🎯 Setup finished. Run without --setup-only to start dashboard")
        return 0
    
    # Step 6: Run dashboard
    if args.production:
        print("🏭 Production mode - dashboard will run on all interfaces")
        return run_dashboard(args.port, "0.0.0.0")
    else:
        print("🧪 Development mode - dashboard will run locally")
        return run_dashboard(args.port, args.host)

if __name__ == "__main__":
    exit(main())
