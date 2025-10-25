# FILE 8: auto_git_sync.py
# Location: (project root) / auto_git_sync.py
# Automatic Git sync script - Commits and pushes changes every 5 minutes
# Run with: python auto_git_sync.py

#!/usr/bin/env python3
import time
import subprocess
import os
from datetime import datetime
import sys

class GitAutoSync:
    def __init__(self, interval_seconds=300):
        """
        Initialize auto-sync
        
        Args:
            interval_seconds: Time between syncs (default: 300 = 5 minutes)
        """
        self.interval = interval_seconds
        self.sync_count = 0
        
    def check_git_repo(self):
        """Check if current directory is a git repository"""
        try:
            subprocess.run(['git', 'status'], 
                         capture_output=True, 
                         check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def has_changes(self):
        """Check if there are any changes to commit"""
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, 
                                  text=True)
            return len(result.stdout.strip()) > 0
        except subprocess.CalledProcessError:
            return False
    
    def git_sync(self):
        """Sync changes with GitHub"""
        try:
            # Check for changes first
            if not self.has_changes():
                print(f"ℹ️  No changes to sync at {datetime.now().strftime('%H:%M:%S')}")
                return True
            
            # Add all changes
            subprocess.run(['git', 'add', '.'], 
                         check=True,
                         capture_output=True)
            
            # Commit with timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f"Auto-sync: {timestamp}"
            
            result = subprocess.run(['git', 'commit', '-m', message], 
                                  capture_output=True,
                                  text=True)
            
            # Check if commit was successful
            if result.returncode != 0:
                if "nothing to commit" in result.stdout:
                    print(f"ℹ️  Nothing to commit at {timestamp}")
                    return True
                else:
                    print(f"⚠️  Commit warning: {result.stderr}")
                    return False
            
            # Push to GitHub
            push_result = subprocess.run(['git', 'push'], 
                                       capture_output=True,
                                       text=True,
                                       timeout=30)
            
            if push_result.returncode == 0:
                self.sync_count += 1
                print(f"✅ Sync #{self.sync_count} completed at {timestamp}")
                return True
            else:
                print(f"❌ Push failed: {push_result.stderr}")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"❌ Git error: {e}")
            return False
        except subprocess.TimeoutExpired:
            print("⏰ Push timeout - check your internet connection")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def start(self):
        """Start auto-sync loop"""
        print("=" * 60)
        print("🔄 Git Auto-Sync Started")
        print("=" * 60)
        print(f"📁 Directory: {os.getcwd()}")
        print(f"⏱️  Sync interval: {self.interval} seconds ({self.interval//60} minutes)")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 60)
        print()
        
        # Check if this is a git repo
        if not self.check_git_repo():
            print("❌ Error: This is not a git repository!")
            print("   Run 'git init' first to initialize git.")
            return
        
        # Run initial sync
        print("🔄 Running initial sync...")
        self.git_sync()
        print()
        
        # Main loop
        try:
            while True:
                print(f"⏳ Next sync in {self.interval} seconds...", end='\r')
                time.sleep(self.interval)
                print(" " * 50, end='\r')  # Clear line
                self.git_sync()
                
        except KeyboardInterrupt:
            print("\n")
            print("=" * 60)
            print("🛑 Auto-sync stopped by user")
            print(f"📊 Total syncs completed: {self.sync_count}")
            print("=" * 60)
            sys.exit(0)

def main():
    """Main function"""
    # Parse command line arguments
    interval = 300  # Default 5 minutes
    
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
            if interval < 60:
                print("⚠️  Warning: Interval less than 60 seconds not recommended")
        except ValueError:
            print("❌ Error: Invalid interval. Using default (300 seconds)")
    
    # Create and start auto-sync
    syncer = GitAutoSync(interval_seconds=interval)
    syncer.start()

if __name__ == "__main__":
    main()