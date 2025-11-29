# gui.py - GUI-only interface for 25th ITMC Chatbot (RAG)
# Backend: imports chatbot.py from the same "app" folder.

import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re

# Lazy backend import: chatbot will be imported in a background thread
chatbot = None


class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("25th ITMC Chatbot - RAG Assistant")
        self.root.geometry("1100x700")

        # Try to set a window icon if available
        try:
            self.root.iconbitmap("25th_itmc.ico")
        except Exception:
            try:
                self.root.iconbitmap("25th_itmc.ico.ico")
            except Exception:
                pass

        # Backend / thinking state
        self.is_thinking = False
        self.think_start_time = None
        self.backend_ready = False
        self.last_bot_answer = ""
        self.answer_counter = 0  # for per-answer copy tags

        # Colors
        self.color_primary = "#1565C0"
        self.color_primary_light = "#E3F2FD"
        self.color_user_bg = "#E3F2FD"
        self.color_bot_bg = "#FFFFFF"
        self.color_quote_bg = "#E0F2F1"
        self.color_chat_bg = "#FAFAFA"

        # --- ttk styles ---
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        default_font = ("Segoe UI", 10)
        self.root.option_add("*Font", default_font)

        style.configure("Header.TFrame", background=self.color_primary)
        style.configure("Header.TLabel", background=self.color_primary,
                        foreground="white", font=("Segoe UI", 14, "bold"))

        style.configure("Side.TFrame", background=self.color_primary_light)
        style.configure("Side.TLabelframe", background=self.color_primary_light)
        style.configure("Side.TLabelframe.Label", background=self.color_primary_light)

        style.configure("Status.TLabel", background="#FFFFFF", foreground="#2E7D32")
        style.configure("Thinking.TLabel", background="#FFFFFF", foreground="#1565C0")
        style.configure("Warn.TLabel", background="#FFFFFF", foreground="#EF6C00")
        style.configure("Error.TLabel", background="#FFFFFF", foreground="#C62828")

        # --- Main layout ---
        self.root.configure(background="#FFFFFF")
        main_frame = ttk.Frame(root, padding=0)
        main_frame.pack(fill="both", expand=True)

        # Header
        header = ttk.Frame(main_frame, style="Header.TFrame")
        header.pack(fill="x")
        header_label = ttk.Label(
            header,
            text="25th ITMC Chatbot – Policy & Education RAG",
            style="Header.TLabel",
            anchor="w",
            padding=(12, 8),
        )
        header_label.pack(fill="x")

        # Content area
        content_frame = ttk.Frame(main_frame, padding=10)
        content_frame.pack(fill="both", expand=True)

        # Left: controls
        left = tk.Frame(content_frame, bg=self.color_primary_light)
        left.pack(side="left", fill="y", padx=(0, 10))

        # Right: chat
        right = ttk.Frame(content_frame)
        right.pack(side="right", fill="both", expand=True)

        # --- Left panel: KB & options ---
        kb_group = ttk.LabelFrame(left, text="Knowledge Base & Options", padding=8)
        kb_group.configure(style="Side.TLabelframe")
        kb_group.pack(fill="y", expand=False, padx=6, pady=6)

        ttk.Label(kb_group, text="KB Mode:").pack(anchor="w", pady=(0, 2))
        self.kb_mode_var = tk.StringVar(value="full")

        rb_base = ttk.Radiobutton(
            kb_group,
            text="Base (original docs only)",
            variable=self.kb_mode_var,
            value="base",
            command=self.update_kb_mode,
        )
        rb_full = ttk.Radiobutton(
            kb_group,
            text="Full (with uploaded docs)",
            variable=self.kb_mode_var,
            value="full",
            command=self.update_kb_mode,
        )
        rb_base.pack(anchor="w")
        rb_full.pack(anchor="w", pady=(0, 4))

        ttk.Separator(kb_group, orient="horizontal").pack(fill="x", pady=5)

        ttk.Label(kb_group, text="Model:").pack(anchor="w")
        self.model_var = tk.StringVar(value="Auto")
        models = ["Auto", "llama3.2:1b", "llama3.2:3b"]
        self.model_combo = ttk.Combobox(
            kb_group,
            textvariable=self.model_var,
            values=models,
            state="readonly",
        )
        self.model_combo.pack(fill="x", pady=(0, 4))

        self.detailed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            kb_group,
            text="Detailed answer",
            variable=self.detailed_var,
        ).pack(anchor="w")

        ttk.Separator(kb_group, orient="horizontal").pack(fill="x", pady=5)

        ttk.Button(
            kb_group,
            text="📄 Upload file…",
            command=self.upload_file,
        ).pack(fill="x", pady=2)
        ttk.Button(
            kb_group,
            text="📁 Load databank folder…",
            command=self.load_databank,
        ).pack(fill="x", pady=2)

        ttk.Separator(kb_group, orient="horizontal").pack(fill="x", pady=5)

        ttk.Label(kb_group, text="Target file (optional):").pack(anchor="w")
        self.target_file_var = tk.StringVar(value="All files")
        self.file_combo = ttk.Combobox(
            kb_group,
            textvariable=self.target_file_var,
            state="readonly",
        )
        self.file_combo.pack(fill="x", pady=(0, 2))
        ttk.Button(
            kb_group,
            text="🔄 Refresh file list",
            command=self.refresh_file_list,
        ).pack(fill="x", pady=2)

        # --- Right panel: chat area ---
        chat_group = ttk.LabelFrame(right, text="Conversation", padding=8)
        chat_group.pack(fill="both", expand=True)

        chat_frame = ttk.Frame(chat_group)
        chat_frame.pack(fill="both", expand=True)

        self.chat_text = tk.Text(
            chat_frame,
            wrap="word",
            state="normal",   # keep normal so selection/copy work
            background=self.color_chat_bg,
            relief="flat",
            borderwidth=8,
        )
        chat_scroll = ttk.Scrollbar(
            chat_frame, orient="vertical", command=self.chat_text.yview
        )
        self.chat_text.configure(yscrollcommand=chat_scroll.set)
        self.chat_text.pack(side="left", fill="both", expand=True)
        chat_scroll.pack(side="right", fill="y")

        # Block editing keys but allow navigation + copy/select (Ctrl+A, Ctrl+C)
        def block_modification(event):
            # state bit 0x4 means Control key
            ctrl = (event.state & 0x4) != 0
            if ctrl and event.keysym in ("c", "C", "a", "A"):
                return None  # allow Ctrl+C, Ctrl+A
            if event.keysym in (
                "Left", "Right", "Up", "Down",
                "Home", "End", "Next", "Prior",
                "Tab"
            ):
                return None  # allow navigation
            # Block everything else that could modify text
            if event.char or event.keysym in ("BackSpace", "Delete", "Return"):
                return "break"
            return None

        self.chat_text.bind("<Key>", block_modification)
        self.chat_text.bind("<<Paste>>", lambda e: "break")
        self.chat_text.bind("<<Cut>>", lambda e: "break")
        self.chat_text.bind("<Button-2>", lambda e: "break")  # block middle-click paste

        # Text tags for formatting and bubbles
        self.chat_text.tag_configure(
            "user_prefix",
            foreground="#0D47A1",
            background=self.color_user_bg,
            font=("Segoe UI", 10, "bold"),
            spacing1=6,
            lmargin1=8,
            lmargin2=8,
            rmargin=80,
        )
        self.chat_text.tag_configure(
            "user_text",
            foreground="black",
            background=self.color_user_bg,
            font=("Segoe UI", 10),
            spacing3=4,
            lmargin1=8,
            lmargin2=8,
            rmargin=80,
        )
        self.chat_text.tag_configure(
            "system_text",
            foreground="#37474F",
            background="#ECEFF1",
            font=("Segoe UI", 9, "italic"),
            lmargin1=6,
            lmargin2=6,
            rmargin=40,
            spacing1=4,
            spacing3=4,
        )
        self.chat_text.tag_configure(
            "bot_prefix",
            foreground="#1B5E20",
            background=self.color_bot_bg,
            font=("Segoe UI", 10, "bold"),
            spacing1=6,
            lmargin1=8,
            lmargin2=8,
            rmargin=80,
        )
        self.chat_text.tag_configure(
            "bot_heading1",
            foreground="#1B5E20",
            background=self.color_bot_bg,
            font=("Segoe UI", 14, "bold"),
            spacing1=6,
            spacing3=4,
            lmargin1=12,
            lmargin2=12,
            rmargin=80,
        )
        self.chat_text.tag_configure(
            "bot_heading2",
            foreground="#2E7D32",
            background=self.color_bot_bg,
            font=("Segoe UI", 11, "bold"),
            spacing1=4,
            spacing3=2,
            lmargin1=12,
            lmargin2=12,
            rmargin=80,
        )
        self.chat_text.tag_configure(
            "bot_bullet",
            foreground="black",
            background=self.color_bot_bg,
            font=("Segoe UI", 10),
            lmargin1=24,
            lmargin2=40,
            rmargin=80,
            spacing1=1,
            spacing3=1,
        )
        self.chat_text.tag_configure(
            "bot_numbered",
            foreground="black",
            background=self.color_bot_bg,
            font=("Segoe UI", 10),
            lmargin1=24,
            lmargin2=40,
            rmargin=80,
            spacing1=1,
            spacing3=1,
        )
        self.chat_text.tag_configure(
            "bot_quote",
            foreground="#455A64",
            background=self.color_quote_bg,
            font=("Segoe UI", 10, "italic"),
            lmargin1=16,
            lmargin2=16,
            rmargin=80,
            spacing1=2,
            spacing3=2,
        )
        self.chat_text.tag_configure(
            "bot_body",
            foreground="black",
            background=self.color_bot_bg,
            font=("Segoe UI", 10),
            lmargin1=12,
            lmargin2=12,
            rmargin=80,
            spacing1=1,
            spacing3=3,
        )
        self.chat_text.tag_configure(
            "copy_link",
            foreground="#0D47A1",
            background=self.color_bot_bg,
            font=("Segoe UI", 9, "underline"),
            lmargin1=12,
            lmargin2=12,
            rmargin=80,
        )

        # Right-click context menu
        self.chat_menu = tk.Menu(self.chat_text, tearoff=0)
        self.chat_menu.add_command(label="Copy selection", command=self.copy_selection)
        self.chat_menu.add_command(label="Select all", command=self.select_all)
        self.chat_text.bind("<Button-3>", self.show_chat_menu)  # Right-click

        # Input area
        input_frame = ttk.Frame(chat_group)
        input_frame.pack(fill="x", pady=(6, 0))
        self.question_var = tk.StringVar()
        self.entry = ttk.Entry(input_frame, textvariable=self.question_var)
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.entry.bind("<Return>", self.on_ask)
        self.ask_button = ttk.Button(input_frame, text="➤ Ask", command=self.on_ask)
        self.ask_button.pack(side="right")

        copy_frame = ttk.Frame(chat_group)
        copy_frame.pack(fill="x", pady=(4, 0))
        self.copy_last_btn = ttk.Button(
            copy_frame,
            text="📋 Copy last answer",
            command=self.copy_last_answer,
        )
        self.copy_last_btn.pack(side="right")

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill="x")
        self.status_var = tk.StringVar(value="Starting backend…")
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            style="Warn.TLabel",
            anchor="w",
            padding=(8, 2),
        )
        self.status_label.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(
            status_frame, mode="indeterminate", length=180
        )
        self.progress.pack(side="right", padx=(0, 8), pady=4)

        # Initial info + backend init
        self.append_system("GUI started. Initializing backend (knowledge base and models)…")
        self.start_thinking("Initializing backend…", status_style="Warn.TLabel")
        self.start_backend_thread()

    # --- Text helpers / formatting ---

    def append_system(self, text: str):
        self.chat_text.insert("end", f"ℹ System: {text}\n\n", ("system_text",))
        self.chat_text.see("end")

    def append_user(self, text: str):
        self.chat_text.insert("end", "🧑 You:\n", ("user_prefix",))
        self.chat_text.insert("end", f"{text}\n\n", ("user_text",))
        self.chat_text.see("end")

    def render_bot_answer(self, text: str):
        """Render bot answer; add per-answer copy link."""
        self.answer_counter += 1
        ans_tag = f"answer_{self.answer_counter}"
        link_tag = f"copy_link_{self.answer_counter}"

        # Mark start of this answer block
        start_index = self.chat_text.index("end")

        self.chat_text.insert("end", "🤖 Bot:\n", ("bot_prefix",))

        first_content_line_seen = False
        paragraphs = [p for p in text.split("\n\n") if p.strip()]

        for para in paragraphs:
            lines = para.splitlines()
            for line in lines:
                raw = line.rstrip()
                s = raw.strip()
                if not s:
                    continue

                s_clean = s.replace("**", "")
                raw_clean = raw.replace("**", "")

                # First non-empty content line -> main heading
                if not first_content_line_seen:
                    self.chat_text.insert("end", raw_clean + "\n", ("bot_heading1",))
                    first_content_line_seen = True
                    continue

                # Lines ending with ":" and not already a list/heading -> subheading
                if (
                    s_clean.endswith(":")
                    and not s_clean.startswith(("#", ">", "-", "*", "•"))
                    and not re.match(r"^\d+[\.\)]\s+", s_clean)
                ):
                    self.chat_text.insert("end", raw_clean + "\n", ("bot_heading2",))
                    continue

                # Markdown headings
                if s_clean.startswith("##"):
                    content = s_clean.lstrip("#").strip()
                    self.chat_text.insert("end", content + "\n", ("bot_heading2",))
                elif s_clean.startswith("#"):
                    content = s_clean.lstrip("#").strip()
                    self.chat_text.insert("end", content + "\n", ("bot_heading1",))

                # Quote
                elif s_clean.startswith(">"):
                    content = s_clean.lstrip(">").strip()
                    self.chat_text.insert("end", content + "\n", ("bot_quote",))

                # Bullets
                elif s_clean.startswith(("-", "*", "•")):
                    content = s_clean.lstrip("-*•").strip()
                    self.chat_text.insert("end", "• " + content + "\n", ("bot_bullet",))

                # Numbered list: "1. ..." or "2) ..."
                elif re.match(r"^\d+[\.\)]\s+", s_clean):
                    self.chat_text.insert("end", raw_clean + "\n", ("bot_numbered",))

                # Normal text
                else:
                    self.chat_text.insert("end", raw_clean + "\n", ("bot_body",))

            self.chat_text.insert("end", "\n")

        # Mark end of answer
        end_index = self.chat_text.index("end")
        self.chat_text.tag_add(ans_tag, start_index, end_index)

        # Add clickable "copy this answer" line
        self.chat_text.insert("end", "📋 Copy this answer\n\n", ("copy_link", link_tag))
        self.chat_text.tag_bind(
            link_tag,
            "<Button-1>",
            lambda e, t=ans_tag: self.copy_answer_by_tag(t)
        )

        self.chat_text.see("end")

    # --- Copy / context menu ---

    def copy_answer_by_tag(self, tag_name: str):
        ranges = self.chat_text.tag_ranges(tag_name)
        if not ranges:
            return
        start, end = ranges[0], ranges[1]
        text = self.chat_text.get(start, end)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.append_system("Answer copied to clipboard.")

    def show_chat_menu(self, event):
        try:
            self.chat_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.chat_menu.grab_release()

    def copy_selection(self):
        try:
            selection = self.chat_text.get("sel.first", "sel.last")
        except tk.TclError:
            self.append_system("No text selected to copy.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(selection)
        self.append_system("Selection copied to clipboard.")

    def select_all(self):
        self.chat_text.tag_add("sel", "1.0", "end")
        self.chat_text.mark_set("insert", "1.0")
        self.chat_text.see("end")

    def copy_last_answer(self):
        if not self.last_bot_answer:
            self.append_system("No bot answer to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_bot_answer)
        self.append_system("Last answer copied to clipboard.")

    # --- Status / thinking ---

    def set_status(self, msg, style_name=None):
        if style_name:
            self.status_label.configure(style=style_name)
        self.status_var.set(msg)

    def start_thinking(self, msg="Thinking…", status_style="Thinking.TLabel"):
        self.is_thinking = True
        self.think_start_time = time.time()
        self.ask_button.configure(state="disabled")
        self.entry.configure(state="disabled")
        self.copy_last_btn.configure(state="disabled")
        self.progress.start(80)
        self.set_status(msg, status_style)
        self.update_elapsed_time()

    def stop_thinking(self, msg="Ready", status_style="Status.TLabel"):
        self.is_thinking = False
        self.progress.stop()
        if self.backend_ready:
            self.ask_button.configure(state="normal")
            self.entry.configure(state="normal")
            self.copy_last_btn.configure(state="normal")
        self.set_status(msg, status_style)

    def update_elapsed_time(self):
        if not self.is_thinking or not self.think_start_time:
            return
        elapsed = int(time.time() - self.think_start_time)
        self.status_var.set(f"Thinking… ({elapsed} s)")
        self.root.after(1000, self.update_elapsed_time)

    # --- Backend init / actions ---

    def start_backend_thread(self):
        threading.Thread(target=self._backend_init, daemon=True).start()

    def _backend_init(self):
        global chatbot
        try:
            import chatbot as cb
            chatbot = cb
            self.backend_ready = True
            self.kb_mode_var.set(getattr(chatbot, "KB_MODE", "full"))
            self.refresh_file_list()
            self.append_system("Backend initialized. Knowledge base and models are ready.")
            self.set_status("Ready", "Status.TLabel")
        except Exception as e:
            self.append_system(f"Error initializing backend: {e}")
            self.set_status("Error during backend initialization", "Error.TLabel")
        finally:
            self.stop_thinking(self.status_var.get())

    def update_kb_mode(self):
        if not self.backend_ready or chatbot is None:
            self.append_system("Backend is still initializing; KB mode cannot be changed yet.")
            return
        mode = self.kb_mode_var.get()
        chatbot.KB_MODE = mode
        self.append_system(f"KB mode set to {mode.upper()}")

    def refresh_file_list(self):
        try:
            if not self.backend_ready or chatbot is None:
                self.file_combo["values"] = ["All files"]
                self.target_file_var.set("All files")
                return
            files = sorted({
                c.get("source_file", "")
                for c in getattr(chatbot, "chunks", [])
                if c.get("source_file")
            })
            values = ["All files"] + files
            self.file_combo["values"] = values
            if self.target_file_var.get() not in values:
                self.target_file_var.set("All files")
        except Exception as e:
            messagebox.showerror("Error", f"Could not refresh file list:\n{e}")

    def upload_file(self):
        if not self.backend_ready or chatbot is None:
            self.append_system("Please wait… backend is still initializing.")
            return
        path = filedialog.askopenfilename(
            title="Select document",
            filetypes=[
                ("Supported files", "*.pdf;*.docx;*.txt;*.xlsx;*.xls"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.append_system(f"Uploading file: {path}")
        self.start_thinking("Uploading and indexing file…", "Thinking.TLabel")
        threading.Thread(target=self._upload_file_thread, args=(path,), daemon=True).start()

    def _upload_file_thread(self, path):
        try:
            res = chatbot.process_file(path, persist=True)
            if res:
                self.append_system(f"File added with {len(res)} chunks.")
                self.refresh_file_list()
            else:
                self.append_system("File was not added (see console/logs if available).")
        except Exception as e:
            self.append_system(f"Error during upload: {e}")
            self.set_status("Upload error", "Error.TLabel")
        finally:
            self.stop_thinking()

    def load_databank(self):
        if not self.backend_ready or chatbot is None:
            self.append_system("Please wait… backend is still initializing.")
            return
        folder = filedialog.askdirectory(title="Select databank folder")
        if not folder:
            return
        self.append_system(f"Loading databank from: {folder}")
        self.start_thinking("Loading databank (this may take a while)…", "Thinking.TLabel")
        threading.Thread(target=self._load_databank_thread, args=(folder,), daemon=True).start()

    def _load_databank_thread(self, folder):
        try:
            chatbot.load_databank(folder, persist=True)
            self.append_system("Databank loading finished.")
            self.refresh_file_list()
        except Exception as e:
            self.append_system(f"Error during databank load: {e}")
            self.set_status("Databank error", "Error.TLabel")
        finally:
            self.stop_thinking()

    def on_ask(self, event=None):
        if not self.backend_ready or chatbot is None:
            self.append_system("Please wait… backend is still initializing.")
            return
        question = self.question_var.get().strip()
        if not question:
            return
        self.question_var.set("")
        self.append_user(question)
        self.start_thinking("Thinking about your question…", "Thinking.TLabel")
        threading.Thread(target=self._ask_thread, args=(question,), daemon=True).start()

    def _ask_thread(self, question):
        try:
            target = self.target_file_var.get()
            target_file = None if target == "All files" else target
            model_sel = self.model_var.get()
            force_model = None if model_sel == "Auto" else model_sel
            detailed = self.detailed_var.get()
            answer = chatbot.query_rag(
                question,
                detailed=detailed,
                target_file=target_file,
                force_model=force_model,
            )
            if not answer:
                answer = "(No answer returned – see documentation or try a simpler question.)"
            self.last_bot_answer = answer.strip()
            self.render_bot_answer(self.last_bot_answer)
        except Exception as e:
            self.append_system(f"Error while answering: {e}")
            self.set_status("Answer error", "Error.TLabel")
        finally:
            self.stop_thinking()


if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()
