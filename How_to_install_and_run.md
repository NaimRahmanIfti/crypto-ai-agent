# 🚀 HOW TO INSTALL AND RUN - AI CRYPTO TRADER

**Complete Installation Guide for Any Computer**

This guide will help you install and run the AI Crypto Trader on **Windows**, **macOS**, or **Linux** - even if you've never coded before!

---

## 📋 **TABLE OF CONTENTS**

1. [System Requirements](#system-requirements)
2. [Windows Installation](#windows-installation)
3. [macOS Installation](#macos-installation)
4. [Linux Installation](#linux-installation)
5. [Verify Installation](#verify-installation)
6. [Running the Application](#running-the-application)
7. [Troubleshooting](#troubleshooting)
8. [Common Issues](#common-issues)

---

## 💻 **SYSTEM REQUIREMENTS**

### **Minimum Requirements:**
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **RAM**: 4GB (8GB recommended)
- **Disk Space**: 2GB free space
- **Internet**: Active internet connection
- **Python**: 3.8 or higher

### **What You'll Install:**
- Python 3.8+ (if not installed)
- pip (Python package manager)
- Virtual environment (optional but recommended)
- 8 Python libraries (~500MB)

---

## 🪟 **WINDOWS INSTALLATION**

### **Step 1: Check if Python is Installed**

1. Press `Win + R` keys
2. Type `cmd` and press Enter
3. In the black window (Command Prompt), type:
   ```cmd
   python --version
   ```
4. If you see `Python 3.8` or higher → **Great! Skip to Step 3**
5. If you see an error → **Continue to Step 2**

### **Step 2: Install Python (if needed)**

1. **Go to:** https://www.python.org/downloads/
2. **Click:** "Download Python 3.12" (big yellow button)
3. **Run** the downloaded file
4. **⚠️ IMPORTANT:** Check the box "Add Python to PATH"
5. Click "Install Now"
6. Wait for installation (2-3 minutes)
7. Click "Close"

### **Step 3: Download the Project**

**Option A: Download ZIP**
1. Download all project files to your Desktop
2. Right-click → "Extract All"
3. Extract to `C:\Users\YourName\Desktop\ai-crypto-trader`

**Option B: Use Git (if you have it)**
```cmd
cd Desktop
git clone https://github.com/yourusername/ai-crypto-trader.git
cd ai-crypto-trader
```

### **Step 4: Open Command Prompt in Project Folder**

1. Open File Explorer
2. Navigate to `Desktop\ai-crypto-trader`
3. Click in the address bar at the top
4. Type `cmd` and press Enter
5. A Command Prompt window opens in that folder

### **Step 5: Create Virtual Environment (Recommended)**

```cmd
python -m venv venv
```

Wait 30 seconds for it to create the environment.

### **Step 6: Activate Virtual Environment**

```cmd
venv\Scripts\activate
```

You should see `(venv)` appear at the start of the line.

### **Step 7: Install Dependencies**

```cmd
pip install numpy pandas ccxt ta scikit-learn tensorflow streamlit plotly python-dateutil
```

This will take **5-10 minutes**. You'll see lots of text scrolling by - this is normal!

**Alternative (if you have requirements.txt):**
```cmd
pip install -r requirements.txt
```

### **Step 8: Run the Application**

```cmd
streamlit run app_trading_COMPLETE.py
```

Your browser will automatically open to `http://localhost:8501`

🎉 **Success! The app is running!**

### **To Stop the Application:**
Press `Ctrl + C` in the Command Prompt

### **To Run Again Later:**

```cmd
cd Desktop\ai-crypto-trader
venv\Scripts\activate
streamlit run app_trading_COMPLETE.py
```

---

## 🍎 **macOS INSTALLATION**

### **Step 1: Check if Python is Installed**

1. Press `Cmd + Space` (opens Spotlight)
2. Type `terminal` and press Enter
3. In Terminal, type:
   ```bash
   python3 --version
   ```
4. If you see `Python 3.8` or higher → **Great! Skip to Step 3**
5. If you see an error → **Continue to Step 2**

### **Step 2: Install Python (if needed)**

**Option A: Using Homebrew (Recommended)**

1. Install Homebrew first (if you don't have it):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```bash
   brew install python@3.12
   ```

**Option B: Download Installer**

1. **Go to:** https://www.python.org/downloads/macos/
2. **Download:** "macOS 64-bit installer"
3. **Run** the downloaded .pkg file
4. Follow the installation wizard
5. Enter your password when asked

### **Step 3: Download the Project**

**Option A: Download ZIP**
1. Download all project files
2. Extract to your Desktop
3. You should have: `~/Desktop/ai-crypto-trader`

**Option B: Use Git**
```bash
cd ~/Desktop
git clone https://github.com/yourusername/ai-crypto-trader.git
cd ai-crypto-trader
```

### **Step 4: Open Terminal in Project Folder**

```bash
cd ~/Desktop/ai-crypto-trader
```

### **Step 5: Create Virtual Environment (Recommended)**

```bash
python3 -m venv venv
```

Wait 30 seconds.

### **Step 6: Activate Virtual Environment**

```bash
source venv/bin/activate
```

You should see `(venv)` appear at the start of the line.

### **Step 7: Install Dependencies**

```bash
pip install numpy pandas ccxt ta scikit-learn tensorflow streamlit plotly python-dateutil
```

This takes **5-10 minutes**.

**If you get "permission denied" errors, add `--break-system-packages`:**
```bash
pip install --break-system-packages numpy pandas ccxt ta scikit-learn tensorflow streamlit plotly python-dateutil
```

**Alternative (if you have requirements.txt):**
```bash
pip install -r requirements.txt
```

### **Step 8: Run the Application**

```bash
streamlit run app_trading_COMPLETE.py
```

Your browser will automatically open to `http://localhost:8501`

🎉 **Success! The app is running!**

### **To Stop the Application:**
Press `Ctrl + C` in Terminal

### **To Run Again Later:**

```bash
cd ~/Desktop/ai-crypto-trader
source venv/bin/activate
streamlit run app_trading_COMPLETE.py
```

---

## 🐧 **LINUX INSTALLATION**

### **Step 1: Check if Python is Installed**

Open Terminal and type:
```bash
python3 --version
```

If you see `Python 3.8` or higher → **Great! Skip to Step 3**

### **Step 2: Install Python (if needed)**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Fedora/RHEL:**
```bash
sudo dnf install python3 python3-pip
```

**Arch Linux:**
```bash
sudo pacman -S python python-pip
```

### **Step 3: Download the Project**

**Option A: Download files manually**
```bash
cd ~/Desktop
mkdir ai-crypto-trader
cd ai-crypto-trader
# Download all files here
```

**Option B: Use Git**
```bash
cd ~/Desktop
git clone https://github.com/yourusername/ai-crypto-trader.git
cd ai-crypto-trader
```

### **Step 4: Create Virtual Environment**

```bash
python3 -m venv venv
```

### **Step 5: Activate Virtual Environment**

```bash
source venv/bin/activate
```

### **Step 6: Install Dependencies**

```bash
pip install numpy pandas ccxt ta scikit-learn tensorflow streamlit plotly python-dateutil
```

**Alternative:**
```bash
pip install -r requirements.txt
```

### **Step 7: Run the Application**

```bash
streamlit run app_trading_COMPLETE.py
```

Browser opens to `http://localhost:8501`

🎉 **Success!**

### **To Stop:**
Press `Ctrl + C`

### **To Run Again:**
```bash
cd ~/Desktop/ai-crypto-trader
source venv/bin/activate
streamlit run app_trading_COMPLETE.py
```

---

## ✅ **VERIFY INSTALLATION**

After installing, verify everything works:

### **Test 1: Check Python**
```bash
python --version  # Windows
python3 --version  # macOS/Linux
```
Should show: `Python 3.8` or higher

### **Test 2: Check pip**
```bash
pip --version
```
Should show pip version and Python version

### **Test 3: Check Libraries**
```bash
pip list
```
Should show: numpy, pandas, tensorflow, streamlit, etc.

### **Test 4: Test Import**
```bash
python -c "import tensorflow; import streamlit; print('All good!')"
```
Should print: `All good!`

---

## 🎮 **RUNNING THE APPLICATION**

### **First Time Setup:**

1. **Navigate to project folder**
   - Windows: `cd C:\Users\YourName\Desktop\ai-crypto-trader`
   - macOS/Linux: `cd ~/Desktop/ai-crypto-trader`

2. **Activate virtual environment**
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. **Run the app**
   ```bash
   streamlit run app_trading_COMPLETE.py
   ```

4. **Browser opens automatically** to `http://localhost:8501`

5. **Configure in the UI:**
   - Select exchange (Binance/Kraken)
   - Choose trading pair (BTC/USDT)
   - Pick timeframe (5m recommended)
   - Click "TRAIN & INITIALIZE"

6. **Wait 2-3 minutes** for training

7. **Start trading!** (paper trading first!)

### **Every Time After:**

Just run these 3 commands:

**Windows:**
```cmd
cd Desktop\ai-crypto-trader
venv\Scripts\activate
streamlit run app_trading_COMPLETE.py
```

**macOS/Linux:**
```bash
cd ~/Desktop/ai-crypto-trader
source venv/bin/activate
streamlit run app_trading_COMPLETE.py
```

### **To Stop:**
Press `Ctrl + C` in the terminal

---

## 🐛 **TROUBLESHOOTING**

### **Problem: "python is not recognized"**

**Windows:**
1. Search for "Environment Variables" in Start menu
2. Click "Environment Variables"
3. Under "System Variables", find "Path"
4. Click "Edit"
5. Add: `C:\Users\YourName\AppData\Local\Programs\Python\Python312`
6. Click OK
7. Close and reopen Command Prompt

**macOS/Linux:**
Use `python3` instead of `python`

### **Problem: "Permission denied"**

**macOS/Linux:**
```bash
pip install --user package-name
# OR
pip install --break-system-packages package-name
```

**Windows:**
Run Command Prompt as Administrator

### **Problem: "No module named 'streamlit'"**

The virtual environment is not activated or libraries not installed.

**Solution:**
```bash
# Activate venv first
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Then install
pip install streamlit
```

### **Problem: "Port 8501 is already in use"**

Another instance is running.

**Solution:**
```bash
# Kill the process
# Windows:
taskkill /F /IM streamlit.exe

# macOS/Linux:
pkill -f streamlit

# Or use different port:
streamlit run app_trading_COMPLETE.py --server.port 8502
```

### **Problem: TensorFlow installation fails**

**For CPU-only (smaller, faster install):**
```bash
pip install tensorflow-cpu
```

**For M1/M2 Mac:**
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

### **Problem: "Illegal instruction" or crash on old CPU**

Your CPU doesn't support AVX instructions.

**Solution:**
```bash
# Use CPU-only version
pip uninstall tensorflow
pip install tensorflow-cpu
```

### **Problem: Very slow training**

**Solutions:**
1. Reduce episodes (50 → 25)
2. Use smaller dataset (limit: 100)
3. Use tensorflow-cpu (sometimes faster on laptops)
4. Close other programs

### **Problem: Exchange API errors**

**Solutions:**
1. Check internet connection
2. Try different exchange
3. Check if exchange is under maintenance
4. Wait a few minutes and retry

---

## 🔧 **COMMON ISSUES**

### **Issue: ModuleNotFoundError**

**Cause:** Library not installed or wrong environment

**Fix:**
```bash
# Make sure venv is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install missing library
pip install library-name
```

### **Issue: Browser doesn't open automatically**

**Fix:**
Manually open browser and go to: `http://localhost:8501`

### **Issue: Application freezes during training**

**Normal behavior!** Training takes 2-3 minutes. Watch the progress bar.

**If truly frozen (>10 minutes):**
- Press `Ctrl + C`
- Reduce episodes to 25
- Try again

### **Issue: Out of memory**

**Fix:**
```python
# Edit app_trading_COMPLETE.py
# Change these values:
MEMORY_SIZE = 1000  # Instead of 2000
BATCH_SIZE = 16     # Instead of 32
```

### **Issue: Charts not displaying**

**Fix:**
```bash
pip install --upgrade plotly
```

### **Issue: Can't find app_trading_COMPLETE.py**

**Fix:**
Make sure you're in the correct folder:
```bash
# Check current directory
pwd         # macOS/Linux
cd          # Windows

# List files
ls          # macOS/Linux
dir         # Windows

# Should see app_trading_COMPLETE.py
```

---

## 📝 **QUICK REFERENCE COMMANDS**

### **Windows Quick Start:**
```cmd
cd Desktop\ai-crypto-trader
venv\Scripts\activate
streamlit run app_trading_COMPLETE.py
```

### **macOS/Linux Quick Start:**
```bash
cd ~/Desktop/ai-crypto-trader
source venv/bin/activate
streamlit run app_trading_COMPLETE.py
```

### **Install Missing Library:**
```bash
pip install library-name
```

### **Update Library:**
```bash
pip install --upgrade library-name
```

### **Check Installed Libraries:**
```bash
pip list
```

### **Deactivate Virtual Environment:**
```bash
deactivate
```

---

## 🎓 **VIDEO TUTORIAL (Step-by-Step)**

If you prefer video instructions:

1. **Search YouTube:** "How to install Python [Your OS]"
2. **Search YouTube:** "How to use pip Python"
3. **Search YouTube:** "Python virtual environment tutorial"

---

## 💡 **PRO TIPS**

### **Tip 1: Create Desktop Shortcut**

**Windows:**
Create a `.bat` file on Desktop:
```batch
@echo off
cd C:\Users\YourName\Desktop\ai-crypto-trader
call venv\Scripts\activate
streamlit run app_trading_COMPLETE.py
pause
```

**macOS/Linux:**
Create a `.sh` file on Desktop:
```bash
#!/bin/bash
cd ~/Desktop/ai-crypto-trader
source venv/bin/activate
streamlit run app_trading_COMPLETE.py
```
Make it executable: `chmod +x run.sh`

### **Tip 2: Auto-Start Browser**

The app automatically opens your browser. If not:
```bash
streamlit run app_trading_COMPLETE.py --browser.serverAddress localhost
```

### **Tip 3: Run on Different Port**

```bash
streamlit run app_trading_COMPLETE.py --server.port 8502
```

### **Tip 4: Keep Terminal Open**

Don't close the terminal window while using the app!

### **Tip 5: Save Terminal Commands**

Save commonly used commands in a text file for quick copy-paste.

---

## 🆘 **STILL HAVING ISSUES?**

### **Step 1: Check Installation**
```bash
# Verify Python
python --version

# Verify pip
pip --version

# Verify libraries
pip list | grep streamlit
pip list | grep tensorflow
```

### **Step 2: Try Clean Install**
```bash
# Remove virtual environment
rm -rf venv  # macOS/Linux
rmdir /s venv  # Windows

# Recreate and reinstall
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Step 3: Check Requirements**
- Python 3.8 or higher ✓
- Internet connection ✓
- 2GB free disk space ✓
- Administrator/sudo access ✓

### **Step 4: Try Alternative Python**
```bash
# Use Python 3.9 or 3.10 instead of 3.12
# Download from python.org
```

### **Step 5: Get Help**
- Read error messages carefully
- Copy exact error to Google
- Check GitHub Issues
- Ask in community forums

---

## 📞 **SUPPORT RESOURCES**

- **Python Installation:** https://www.python.org/downloads/
- **pip Documentation:** https://pip.pypa.io/
- **Streamlit Docs:** https://docs.streamlit.io/
- **TensorFlow Install:** https://www.tensorflow.org/install

---

## ✅ **INSTALLATION CHECKLIST**

Before running, make sure:

- [ ] Python 3.8+ installed
- [ ] pip installed and working
- [ ] Project files downloaded
- [ ] Virtual environment created
- [ ] Virtual environment activated
- [ ] All libraries installed (8 total)
- [ ] app_trading_COMPLETE.py exists in folder
- [ ] Internet connection active

---

## 🎉 **SUCCESS!**

If you can see the Streamlit UI in your browser, **congratulations!**

You're ready to:
1. ✅ Configure settings
2. ✅ Train the RL agent
3. ✅ View predictions
4. ✅ Start trading!

---

## 📝 **NEXT STEPS**

1. **Read the README.md** for usage instructions
2. **Start with paper trading** (no real money)
3. **Test for 1-2 weeks** before live trading
4. **Read the documentation** to understand the system

---

**Remember: Start with paper trading and small amounts!**

**Happy Trading! 🚀📈**

---

**Last Updated:** December 2025  
**Version:** 1.0  
**Tested on:** Windows 10/11, macOS 12+, Ubuntu 20.04+
