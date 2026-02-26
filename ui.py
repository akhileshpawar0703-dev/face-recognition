import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


APP_PATH = Path(__file__).with_name("app.py")


class FaceSecureUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Face Secure Toolkit")
        self.geometry("720x520")

        self.camera_var = tk.StringVar(value="0")
        self.detector_var = tk.StringVar(value="auto")
        self.threshold_var = tk.StringVar(value="-1")
        self.output_var = tk.StringVar(value="")

        self._build()

    def _build(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        cfg = ttk.LabelFrame(root, text="Common Settings", padding=10)
        cfg.pack(fill="x")
        ttk.Label(cfg, text="Camera Index").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(cfg, textvariable=self.camera_var, width=8).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(cfg, text="Detector").grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Combobox(cfg, textvariable=self.detector_var, values=["auto", "yunet", "haar"], width=10, state="readonly").grid(row=0, column=3, sticky="w", pady=4)
        ttk.Label(cfg, text="Threshold").grid(row=0, column=4, sticky="w", padx=6, pady=4)
        ttk.Entry(cfg, textvariable=self.threshold_var, width=8).grid(row=0, column=5, sticky="w", pady=4)

        note = ttk.Label(root, text="Tip: Keep threshold = -1 for auto-calibration.")
        note.pack(anchor="w", pady=(6, 8))

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)

        # Run tab
        tab_run = ttk.Frame(nb, padding=12)
        nb.add(tab_run, text="Unlock UI")
        ttk.Button(tab_run, text="Start Face Unlock Window", command=self.start_unlock).pack(anchor="w", pady=8)
        ttk.Label(tab_run, text="This opens the live lock/unlock camera window.").pack(anchor="w")

        # Enroll tab
        tab_enroll = ttk.Frame(nb, padding=12)
        nb.add(tab_enroll, text="Enroll")
        self.enroll_name_var = tk.StringVar()
        self.enroll_samples_var = tk.StringVar(value="12")
        ttk.Label(tab_enroll, text="Person Name").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(tab_enroll, textvariable=self.enroll_name_var, width=28).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Label(tab_enroll, text="Samples").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(tab_enroll, textvariable=self.enroll_samples_var, width=10).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Button(tab_enroll, text="Start Enrollment", command=self.enroll).grid(row=2, column=0, columnspan=2, sticky="w", pady=10)

        # PDF tab
        tab_pdf = ttk.Frame(nb, padding=12)
        nb.add(tab_pdf, text="PDF Lock/Unlock")
        self.pdf_source_var = tk.StringVar()
        self.pdf_owner_var = tk.StringVar()
        self.pdf_enc_var = tk.StringVar()

        ttk.Label(tab_pdf, text="PDF Source").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(tab_pdf, textvariable=self.pdf_source_var, width=52).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Button(tab_pdf, text="Browse", command=self.pick_pdf_source).grid(row=0, column=2, padx=6)

        ttk.Label(tab_pdf, text="Owner Name").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(tab_pdf, textvariable=self.pdf_owner_var, width=28).grid(row=1, column=1, sticky="w", pady=4)
        ttk.Button(tab_pdf, text="Encrypt PDF", command=self.encrypt_pdf).grid(row=2, column=1, sticky="w", pady=8)

        ttk.Separator(tab_pdf, orient="horizontal").grid(row=3, column=0, columnspan=3, sticky="ew", pady=8)

        ttk.Label(tab_pdf, text="Encrypted File").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(tab_pdf, textvariable=self.pdf_enc_var, width=52).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Button(tab_pdf, text="Browse", command=self.pick_pdf_encrypted).grid(row=4, column=2, padx=6)
        ttk.Button(tab_pdf, text="Unlock & Open PDF", command=self.unlock_pdf).grid(row=5, column=1, sticky="w", pady=8)

        # Admin tab
        tab_admin = ttk.Frame(nb, padding=12)
        nb.add(tab_admin, text="Admin")
        self.delete_name_var = tk.StringVar()
        ttk.Label(tab_admin, text="Delete Identity Name").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(tab_admin, textvariable=self.delete_name_var, width=28).grid(row=0, column=1, sticky="w", pady=4)
        ttk.Button(tab_admin, text="Delete Identity", command=self.delete_identity).grid(row=1, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Button(tab_admin, text="Verify Audit Chain", command=self.verify_audit).grid(row=2, column=0, columnspan=2, sticky="w", pady=8)

        out_box = ttk.LabelFrame(root, text="Output", padding=8)
        out_box.pack(fill="both", expand=True, pady=(10, 0))
        self.output = tk.Text(out_box, height=8, wrap="word")
        self.output.pack(fill="both", expand=True)

    def log(self, text: str) -> None:
        self.output.insert("end", text + "\n")
        self.output.see("end")

    def _base_args(self) -> list[str]:
        return [
            sys.executable,
            str(APP_PATH),
            "--camera-index",
            self.camera_var.get().strip() or "0",
            "--detector",
            self.detector_var.get().strip() or "auto",
            "--threshold",
            self.threshold_var.get().strip() or "-1",
        ]

    def run_cmd(self, args: list[str]) -> None:
        self.log("$ " + " ".join(args))
        try:
            res = subprocess.run(args, text=True, capture_output=True)
            if res.stdout:
                self.log(res.stdout.strip())
            if res.stderr:
                self.log(res.stderr.strip())
            if res.returncode == 0:
                self.log("✅ Done")
            else:
                self.log(f"❌ Exit code {res.returncode}")
                messagebox.showerror("Command failed", res.stderr.strip() or "Command failed")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_unlock(self) -> None:
        self.run_cmd(self._base_args())

    def enroll(self) -> None:
        name = self.enroll_name_var.get().strip()
        samples = self.enroll_samples_var.get().strip() or "12"
        if not name:
            messagebox.showwarning("Missing", "Enter person name")
            return
        self.run_cmd(self._base_args() + ["enroll", "--name", name, "--samples", samples])

    def pick_pdf_source(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("PDF", "*.pdf")])
        if p:
            self.pdf_source_var.set(p)

    def pick_pdf_encrypted(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("Face PDF", "*.facepdf"), ("All", "*.*")])
        if p:
            self.pdf_enc_var.set(p)

    def encrypt_pdf(self) -> None:
        src = self.pdf_source_var.get().strip()
        owner = self.pdf_owner_var.get().strip()
        if not src or not owner:
            messagebox.showwarning("Missing", "Select source PDF and owner")
            return
        self.run_cmd(self._base_args() + ["encrypt-pdf", "--pdf", src, "--owner", owner])

    def unlock_pdf(self) -> None:
        enc = self.pdf_enc_var.get().strip()
        if not enc:
            messagebox.showwarning("Missing", "Select encrypted file")
            return
        self.run_cmd(self._base_args() + ["unlock-pdf", "--file", enc])

    def delete_identity(self) -> None:
        name = self.delete_name_var.get().strip()
        if not name:
            messagebox.showwarning("Missing", "Enter identity name")
            return
        self.run_cmd(self._base_args() + ["delete", "--name", name])

    def verify_audit(self) -> None:
        self.run_cmd(self._base_args() + ["verify-audit"])


if __name__ == "__main__":
    FaceSecureUI().mainloop()
