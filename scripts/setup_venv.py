#!/usr/bin/env python3
"""
Virtual Environment Setup Script for SeeSense Dashboard
Creates and configures virtual environment with all dependencies
"""
import subprocess
import sys
import os
from pathlib import Path
import argparse
import venv

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  SeeSense Dashboard requires Python 3.8 or higher")
        print("📥 Please install Python 3.8+ from https://python.org")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment(venv_path, python_executable=None):
    """Create virtual environment"""
    print(f"🔧 Creating virtual environment at {venv_path}")
    
    try:
        if python_executable:
            # Use specific Python executable
            subprocess.run([python_executable, "-m", "venv", str(venv_path)], check=True)
        else:
            # Use current Python
            venv.create(venv_path, with_pip=True)
        
        print(f"✅ Virtual environment created successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def get_activation_command(venv_path):
    """Get the activation command for the current OS"""
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        return f"{activate_script}"
    else:  # Unix/Linux/macOS
        activate_script = venv_path / "bin" / "activate"
        return f"source {activate_script}"

def get_python_executable(venv_path):
    """Get Python executable path in virtual environment"""
    if os.name == 'nt':  # Windows
        return venv_path / "Scripts" / "python.exe"
    else:  # Unix/Linux/macOS
        return venv_path / "bin" / "python"

def install_requirements(venv_path, requirements_file="requirements.txt"):
    """Install requirements in virtual environment"""
    python_exe = get_python_executable(venv_path)
    
    if not python_exe.exists():
        print(f"❌ Python executable not found: {python_exe}")
        return False
    
    print(f"📦 Installing requirements from {requirements_file}")
    
    try:
        # Upgrade pip first
        subprocess.run([
            str(python_exe), "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        # Install requirements
        subprocess.run([
            str(python_exe), "-m", "pip", "install", "-r", requirements_file
        ], check=True)
        
        print("✅ Requirements installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_activation_scripts(venv_path):
    """Create convenient activation scripts"""
    project_root = Path.cwd()
    
    # Create activation script for Unix/Linux/macOS
    if os.name != 'nt':
        activate_script = project_root / "activate_env.sh"
        with open(activate_script, 'w') as f:
            f.write(f"""#!/bin/bash
# SeeSense Dashboard Environment Activation Script

echo "🚲 Activating SeeSense Dashboard environment..."
source {venv_path}/bin/activate

echo "✅ Environment activated!"
echo "🚀 Run the dashboard with: streamlit run app/main.py"
echo "🔧 Generate sample data with: python scripts/generate_sample_data.py"
echo "📊 Deploy with: python scripts/deploy.py"

# Set environment variable
export DASHBOARD_ENV=development

# Change to project directory
cd {project_root}
""")
        
        # Make executable
        os.chmod(activate_script, 0o755)
        print(f"✅ Created activation script: {activate_script}")
    
    # Create activation script for Windows
    if os.name == 'nt':
        activate_script = project_root / "activate_env.bat"
        with open(activate_script, 'w') as f:
            f.write(f"""@echo off
REM SeeSense Dashboard Environment Activation Script

echo 🚲 Activating SeeSense Dashboard environment...
call {venv_path}\\Scripts\\activate.bat

echo ✅ Environment activated!
echo 🚀 Run the dashboard with: streamlit run app/main.py
echo 🔧 Generate sample data with: python scripts/generate_sample_data.py
echo 📊 Deploy with: python scripts/deploy.py

REM Set environment variable
set DASHBOARD_ENV=development

REM Change to project directory
cd /d {project_root}
""")
        print(f"✅ Created activation script: {activate_script}")

def create_deactivation_script():
    """Create deactivation script"""
    project_root = Path.cwd()
    
    if os.name != 'nt':
        deactivate_script = project_root / "deactivate_env.sh"
        with open(deactivate_script, 'w') as f:
            f.write("""#!/bin/bash
# SeeSense Dashboard Environment Deactivation Script

echo "🔄 Deactivating SeeSense Dashboard environment..."
deactivate

echo "✅ Environment deactivated!"
""")
        os.chmod(deactivate_script, 0o755)
        print(f"✅ Created deactivation script: {deactivate_script}")

def create_vscode_settings(venv_path):
    """Create VS Code settings for the virtual environment"""
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    python_exe = get_python_executable(venv_path)
    
    settings = {
        "python.defaultInterpreterPath": str(python_exe),
        "python.terminal.activateEnvironment": True,
        "python.formatting.provider": "black",
        "python.linting.enabled": True,
        "python.linting.flake8Enabled": True,
        "python.linting.mypyEnabled": True,
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            ".venv": True,
            "venv": True
        }
    }
    
    import json
    with open(vscode_dir / "settings.json", 'w') as f:
        json.dump(settings, f, indent=2)
    
    print("✅ Created VS Code settings")

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Set up virtual environment for SeeSense Dashboard")
    parser.add_argument("--name", default=".venv", help="Virtual environment directory name")
    parser.add_argument("--python", help="Specific Python executable to use")
    parser.add_argument("--requirements", default="requirements.txt", 
                       choices=["requirements.txt", "requirements-minimal.txt", "requirements-dev.txt"],
                       help="Requirements file to install")
    parser.add_argument("--no-install", action="store_true", help="Create venv but don't install packages")
    parser.add_argument("--vscode", action="store_true", help="Create VS Code settings")
    
    args = parser.parse_args()
    
    print("🚲 SeeSense Dashboard Virtual Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Set up paths
    venv_path = Path(args.name)
    
    if venv_path.exists():
        response = input(f"⚠️  Directory {venv_path} already exists. Remove and recreate? (y/N): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(venv_path)
            print(f"🗑️  Removed existing {venv_path}")
        else:
            print("❌ Aborted")
            return 1
    
    # Create virtual environment
    if not create_virtual_environment(venv_path, args.python):
        return 1
    
    # Install requirements
    if not args.no_install:
        if not install_requirements(venv_path, args.requirements):
            return 1
    
    # Create activation scripts
    create_activation_scripts(venv_path)
    create_deactivation_script()
    
    # Create VS Code settings
    if args.vscode:
        create_vscode_settings(venv_path)
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 Virtual environment setup complete!")
    print("\n🚀 Next steps:")
    
    activation_cmd = get_activation_command(venv_path)
    
    if os.name == 'nt':  # Windows
        print(f"1. Activate environment: activate_env.bat")
        print(f"   Or manually: {activation_cmd}")
    else:  # Unix/Linux/macOS
        print(f"1. Activate environment: ./activate_env.sh")
        print(f"   Or manually: {activation_cmd}")
    
    print("2. Generate sample data: python scripts/generate_sample_data.py")
    print("3. Run dashboard: streamlit run app/main.py")
    print("4. Deploy: python scripts/deploy.py")
    
    print(f"\n📁 Virtual environment created at: {venv_path.absolute()}")
    print(f"🐍 Python executable: {get_python_executable(venv_path)}")
    print(f"📦 Requirements installed: {args.requirements}")
    
    return 0

if __name__ == "__main__":
    exit(main())
