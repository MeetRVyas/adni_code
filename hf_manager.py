import os
import shutil
import stat
import zipfile
from datetime import datetime
from huggingface_hub import HfApi, create_repo
import smtplib
import ssl
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class HFBackupManager:
    def __init__(self, token, repo_id):
        """
        Args:
            token (str): Your Hugging Face 'Write' token.
            repo_id (str): Target repo (e.g., 'username/project-backups').
        """
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.token = token
        
        # Ensure repo exists as a Dataset (Private by default for safety)
        try:
            create_repo(
                repo_id=self.repo_id, 
                token=self.token, 
                repo_type="dataset", 
                private=True, 
                exist_ok=True
            )
            print(f"✅ Connected to HF Repo: {self.repo_id}")
        except Exception as e:
            print(f"⚠️ Warning: Could not verify repo existence: {e}")

    def _remove_readonly(self, func, path, _):
        """Helper to force delete read-only files (common on Windows/Linux GPU setups)."""
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    def _aggressive_cleanup(self, folder_path):
        """Recursively wipes a directory."""
        print(f"🧹 Cleaning up local folder: {folder_path}")
        if not os.path.exists(folder_path):
            return

        # 1. Try standard fast delete
        try:
            shutil.rmtree(folder_path, onerror=self._remove_readonly)
            # Re-create empty folder so next batch doesn't crash
            os.makedirs(folder_path, exist_ok=True)
            return
        except Exception:
            pass

        # 2. Manual Walk (if standard fails)
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                try:
                    p = os.path.join(root, file)
                    os.chmod(p, stat.S_IWRITE)
                    os.remove(p)
                except Exception: pass
            for d in dirs:
                try:
                    p = os.path.join(root, d)
                    os.rmdir(p)
                except Exception: pass

    def process_batch(self, source_dir, batch_id):
        """
        1. Zips the source_dir.
        2. Uploads Zip to Hugging Face.
        3. Deletes the Zip and empties source_dir.
        """
        if not os.path.exists(source_dir) or not os.listdir(source_dir):
            print("⚠️ Output directory is empty. Skipping backup.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"Batch_{batch_id}_{timestamp}.zip"
        zip_path = os.path.join(os.path.dirname(source_dir), zip_filename)

        print(f"\n[HF Manager] Starting backup for Batch {batch_id}...")

        try:
            # --- STEP 1: ZIP ---
            print(f"📦 Zipping '{source_dir}'...")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Relpath ensures we don't zip the full absolute path
                        arcname = os.path.relpath(file_path, source_dir)
                        zipf.write(file_path, arcname=arcname)
            
            # --- STEP 2: UPLOAD ---
            print(f"🚀 Uploading to Hugging Face: {self.repo_id}...")
            self.api.upload_file(
                path_or_fileobj=zip_path,
                path_in_repo=f"batch_backups/{zip_filename}",
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            print("✅ Upload Successful.")

            # --- STEP 3: CLEANUP ---
            print("🗑️  Performing cleanup...")
            if os.path.exists(zip_path):
                os.remove(zip_path) # Delete the zip
            
            self._aggressive_cleanup(source_dir) # Wipe the results folder
            print(f"[HF Manager] Batch {batch_id} Backup & Cleanup Complete.\n")

        except Exception as e:
            print(f"❌ CRITICAL BACKUP ERROR: {e}")
            # Optional: Don't delete if upload failed, so you don't lose data
    
    def upload_run_summary(self, run_id, content):
        """
        Creates a text file with the run status and uploads it to HF.
        """
        filename = f"REPORT_{run_id}.txt"
        
        try:
            # 1. Write temp file
            with open(filename, "w", encoding = "utf-8") as f:
                f.write(content)
                
            print(f"\n[HF Manager] Uploading Final Report: {filename}...")
            
            # 2. Upload to a specific 'run_logs' folder so it's easy to find
            self.api.upload_file(
                path_or_fileobj=filename,
                path_in_repo=f"run_logs/{filename}",
                repo_id=self.repo_id,
                repo_type="dataset"
            )
            print("✅ Final Report Uploaded.")
            
        except Exception as e:
            print(f"❌ Failed to upload run report: {e}")
            
        # finally:
        #     # 3. Cleanup local file
        #     if os.path.exists(filename):
        #         os.remove(filename)