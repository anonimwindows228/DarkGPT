# Improved main UI
# Developed 19.03.2026
# Thank you very much to Kirill Zinchenko ( @Kflone5 ) For help with this script.

import argparse
import threading
import tkinter as tk
from tkinter import font as tkfont
import torch
from model import TinyLM, ModelConfig

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",   type=str,   default="checkpoints/model.pt")
parser.add_argument("--temp",         type=float, default=0.8)
parser.add_argument("--topk",         type=int,   default=40)
parser.add_argument("--max-tokens",   type=int,   default=400)
parser.add_argument("--model-name",   type=str,   default="DarkGPT v2 Super")
args = parser.parse_args()

# Palette

BG           = "#09090b"
BG_SIDEBAR   = "#08080a"
BG_MSG_USER  = "#18181f"
BG_INPUT     = "#111116"

FG_USER      = "#f4f4f5"
FG_AI        = "#a1a1aa"
FG_LABEL     = "#3f3f46"
FG_DIM       = "#27272a"
FG_SYS       = "#3f3f46"

ACCENT_1     = "#8b5cf6"
ACCENT_2     = "#6366f1"
ACCENT_3     = "#a78bfa"

BORDER       = "#18181b"
BORDER_FOCUS = "#7c3aed"

GREEN        = "#4ade80"
AMBER        = "#fbbf24"
RED          = "#f87171"

# gradient

GH_L = "#0f0a1e"
GH_R = "#0a0f1e"
GB_L = "#6d28d9"
GB_R = "#4f46e5"

# torch

torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    import torch_directml
    device = torch_directml.device()
except Exception:
    pass

model = None; vocab = None; inv_vocab = None
encode = None; decode = None; model_loaded = False


def load_model():
    global model, vocab, inv_vocab, encode, decode, model_loaded
    try:
        ck     = torch.load(args.checkpoint, map_location=device, weights_only=False)
        vocab  = ck["vocab"]; inv_vocab = ck["inv_vocab"]
        encode = lambda s: [vocab.get(c, 0) for c in s]
        decode = lambda l: "".join(inv_vocab.get(i, "?") for i in l)
        model  = TinyLM(ck["config"]).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        model_loaded = True
        return True, ck.get("val_loss", "?"), ck.get("step", "?")
    except FileNotFoundError:
        return False, None, None


def generate_stream(prompt, callback):
    import torch.nn.functional as F
    if not model_loaded:
        callback("Model not loaded.")
        return
    enc = encode(prompt)
    idx = torch.tensor([enc], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(args.max_tokens):
            idx_ctx = idx[:, -model.cfg.context_len:]
            logits, _ = model(idx_ctx)
            logits = logits[:, -1, :] / args.temp
            if args.topk is not None:
                v, _ = torch.topk(logits, min(args.topk, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            idx    = torch.cat([idx, next_t], dim=1)
            callback(inv_vocab.get(next_t.item(), "?"))

def _hex_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def _lerp(c1, c2, t):
    r1,g1,b1=_hex_rgb(c1); r2,g2,b2=_hex_rgb(c2)
    return "#{:02x}{:02x}{:02x}".format(
        int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t))

def _h_grad_img(w, h, c1, c2):
    if w < 1: w = 1
    img = tk.PhotoImage(width=w, height=h)
    r1,g1,b1=_hex_rgb(c1); r2,g2,b2=_hex_rgb(c2)
    cols=["#{:02x}{:02x}{:02x}".format(
        int(r1+(r2-r1)*x/max(w-1,1)),
        int(g1+(g2-g1)*x/max(w-1,1)),
        int(b1+(b2-b1)*x/max(w-1,1))) for x in range(w)]
    row="{" + " ".join(cols) + "}"
    for y in range(h): img.put(row, to=(0,y))
    return img

# Fonts

_FONTS: dict = {}

def _init_fonts():
    fams = tkfont.families()
    sans = next((f for f in ["Segoe UI","SF Pro Display","Helvetica Neue",
                              "Ubuntu","Cantarell","DejaVu Sans"] if f in fams),
                "TkDefaultFont")
    _FONTS["title"]  = (sans, 13, "bold")
    _FONTS["body"]   = (sans, 11)
    _FONTS["small"]  = (sans,  9)
    _FONTS["label"]  = (sans,  8, "bold")
    _FONTS["name"]   = (sans, 10, "bold")

# Gradient canvas

class HGradCanvas(tk.Canvas):
    def __init__(self, parent, c1, c2, height, **kw):
        super().__init__(parent, height=height, highlightthickness=0, bd=0, **kw)
        self._c1, self._c2 = c1, c2
        self._img = None; self._img_id = None
        self.bind("<Configure>", self._redraw)

    def _redraw(self, _e=None):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2: return
        img = _h_grad_img(w, h, self._c1, self._c2)
        self._img = img
        if self._img_id is None:
            self._img_id = self.create_image(0,0,anchor="nw",image=img)
        else:
            self.itemconfig(self._img_id, image=img)
        self.tag_lower(self._img_id)
        self._on_ready()

    def _on_ready(self): pass

# Header

class Header(HGradCanvas):
    def __init__(self, parent, **kw):
        super().__init__(parent, c1=GH_L, c2=GH_R, height=56, **kw)
        self._dot_id = None; self._lbl_id = None

    def _on_ready(self):
        self.delete("ui")
        w, h = self.winfo_width(), self.winfo_height()

        # Glowing
        self.create_oval(20,18,36,34, fill="#1a0a3a", outline=ACCENT_1,
                         width=1, tags="ui")
        self.create_oval(24,22,32,30, fill=ACCENT_1, outline="", tags="ui")
        self.create_oval(26,24,30,28, fill=ACCENT_3, outline="", tags="ui")

        # Title
        self.create_text(46, 28, text=args.model_name,
                         font=_FONTS.get("title",("TkDefaultFont",13,"bold")),
                         fill=FG_USER, anchor="w", tags="ui")

        # Bottom
        self.create_line(0, h-2, w, h-2, fill="#1a0a3a", tags="ui")
        self.create_line(0, h-1, w, h-1, fill="#0f0a1e", tags="ui")

        # Status
        sx = w - 145
        if self._dot_id is None:
            self._dot_id = self.create_oval(sx,22,sx+12,34,
                fill=FG_DIM, outline="", tags="ui")
            self._lbl_id = self.create_text(sx+18, 28, text="Loading",
                font=_FONTS.get("small",("TkDefaultFont",9)),
                fill=FG_DIM, anchor="w", tags="ui")
        else:
            self.coords(self._dot_id, sx,22,sx+12,34)
            self.coords(self._lbl_id, sx+18, 28)

    def set_status(self, label, color):
        if self._dot_id and self._lbl_id:
            self.itemconfig(self._dot_id, fill=color)
            self.itemconfig(self._lbl_id, text=label, fill=color)

# Send button

class SendButton(tk.Canvas):
    W, H = 76, 38

    def __init__(self, parent, command=None, **kw):
        super().__init__(parent, width=self.W, height=self.H,
                         highlightthickness=0, bd=0, cursor="hand2", **kw)
        self._cmd=command; self._disabled=False; self._hover=False
        self._text="Send"; self._img=None
        self._draw()
        self.bind("<Enter>",           lambda _e: self._hset(True))
        self.bind("<Leave>",           lambda _e: self._hset(False))
        self.bind("<ButtonRelease-1>", self._click)

    def _hset(self, v):
        if not self._disabled:
            self._hover=v; self._draw()

    def _draw(self):
        self.delete("all")
        w,h = self.W, self.H
        if self._disabled:
            c1=c2="#1a1a1f"; fg=FG_LABEL
        else:
            c1,c2 = (GB_L,GB_R) if not self._hover else (
                _lerp(GB_L,"#ffffff",0.15), _lerp(GB_R,"#ffffff",0.15))
            fg="#ffffff"
        img = _h_grad_img(w,h,c1,c2)
        self._img=img
        self.create_image(0,0,anchor="nw",image=img)
        self.create_text(w//2, h//2+1, text=self._text,
                         font=_FONTS.get("small",("TkDefaultFont",9,"bold")),
                         fill=fg)

    def _click(self, _e):
        if not self._disabled and self._cmd: self._cmd()

    def set_state(self, state, label=None):
        self._disabled=(state==tk.DISABLED)
        if label: self._text=label
        self._draw()

# Sidebar

class Sidebar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG_SIDEBAR, width=210, **kw)
        self.pack_propagate(False)
        self._pulse_job=None
        self._build()

    def _label(self, parent, text, font_key, fg, **kw):
        return tk.Label(parent, text=text,
                        font=_FONTS.get(font_key,("TkDefaultFont",9)),
                        fg=fg, bg=BG_SIDEBAR, **kw)

    def _section_sep(self):
        tk.Frame(self, bg="#111116", height=1).pack(fill=tk.X, padx=14, pady=(10,0))

    def _build(self):
        # Model identity
        id_frame = tk.Frame(self, bg=BG_SIDEBAR)
        id_frame.pack(fill=tk.X, padx=18, pady=(26,18))

        # Gradient canvas
        av = tk.Canvas(id_frame, width=40, height=40,
                       bg=BG_SIDEBAR, highlightthickness=0)
        av.pack(side=tk.LEFT, padx=(0,12))
        av_img = _h_grad_img(40,40, GB_L, GB_R)
        self._av_img = av_img
        av.create_image(0,0,anchor="nw",image=av_img)
        av.create_text(20,21, text="AI",
                       font=_FONTS.get("name",("TkDefaultFont",10,"bold")),
                       fill="#ffffff")
        av.create_rectangle(0,0,39,39, outline="#2a1060", fill="")

        name_col = tk.Frame(id_frame, bg=BG_SIDEBAR)
        name_col.pack(side=tk.LEFT, fill=tk.X)

        self._label(name_col, args.model_name, "name", FG_USER,
                    anchor="w").pack(anchor="w")
        self._label(name_col, "21.03.2026", "small", FG_LABEL,
                    anchor="w").pack(anchor="w", pady=(2,0))

        self._section_sep()

        # Status
        stat_frame = tk.Frame(self, bg=BG_SIDEBAR)
        stat_frame.pack(fill=tk.X, padx=18, pady=(12,4))

        self._label(stat_frame, "STATUS", "label", FG_LABEL).pack(anchor="w")

        dot_row = tk.Frame(self, bg=BG_SIDEBAR)
        dot_row.pack(fill=tk.X, padx=18, pady=(6,14))
        self._dot_cv = tk.Canvas(dot_row, width=10, height=10,
                                 bg=BG_SIDEBAR, highlightthickness=0)
        self._dot_cv.pack(side=tk.LEFT, padx=(0,8))
        self._dot_oval = self._dot_cv.create_oval(1,1,9,9, fill=FG_DIM, outline="")
        self._status_lbl = self._label(dot_row, "Loading…", "small", FG_AI)
        self._status_lbl.pack(side=tk.LEFT)

        self._section_sep()

        # Model info
        info_frame = tk.Frame(self, bg=BG_SIDEBAR)
        info_frame.pack(fill=tk.X, padx=18, pady=(12,4))
        self._label(info_frame, "MODEL INFO", "label", FG_LABEL).pack(anchor="w")

        self._stat_vars = {}
        for key, default in [("Val loss","—"),("Step","—"),
                              ("Temp", str(args.temp)),
                              ("Max tokens", str(args.max_tokens))]:
            row = tk.Frame(self, bg=BG_SIDEBAR)
            row.pack(fill=tk.X, padx=18, pady=3)
            self._label(row, key, "small", FG_LABEL,
                        width=10, anchor="w").pack(side=tk.LEFT)
            v = tk.StringVar(value=default)
            self._stat_vars[key] = v
            tk.Label(row, textvariable=v,
                     font=_FONTS.get("small",("TkDefaultFont",9)),
                     fg=FG_AI, bg=BG_SIDEBAR, anchor="w").pack(side=tk.LEFT)

        # Spacer
        tk.Frame(self, bg=BG_SIDEBAR).pack(fill=tk.BOTH, expand=True)

        self._section_sep()
        self._label(self, "↵  Send   ·   ⇧↵  New line",
                    "small", FG_DIM, pady=12).pack()

    def set_status(self, text, color):
        self._dot_cv.itemconfig(self._dot_oval, fill=color)
        self._status_lbl.config(text=text, fg=color if color!=AMBER else FG_AI)

    def set_model_info(self, val_loss, step):
        self._stat_vars["Val loss"].set(
            f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss))
        self._stat_vars["Step"].set(
            f"{step:,}" if isinstance(step, int) else str(step))

    def pulse(self):
        self._p_t=0.0; self._p_dir=1
        self._do_pulse()

    def _do_pulse(self):
        self._p_t = max(0.0, min(1.0, self._p_t + 0.07*self._p_dir))
        if self._p_t>=1.0: self._p_dir=-1
        if self._p_t<=0.0: self._p_dir= 1
        col = _lerp(BG_SIDEBAR, AMBER, self._p_t)
        self._dot_cv.itemconfig(self._dot_oval, fill=col)
        self._pulse_job = self.after(45, self._do_pulse)

    def stop_pulse(self):
        if self._pulse_job:
            self.after_cancel(self._pulse_job)
            self._pulse_job=None

# Main

class DarkGPTApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DarkGPT")
        self.root.configure(bg=BG)
        self.root.geometry("1040x700")
        self.root.minsize(700, 500)
        self._generating = False
        self._think_job  = None
        _init_fonts()
        self._build_ui()
        self._load_model_async()

    # Build

    def _build_ui(self):
        self.sidebar = Sidebar(self.root)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        tk.Frame(self.root, bg="#111116", width=1).pack(side=tk.LEFT, fill=tk.Y)

        main = tk.Frame(self.root, bg=BG)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Header
        self.header = Header(main, bg=GH_L)
        self.header.pack(fill=tk.X)

        self._build_input(main)
        self._build_chat(main)

    def _build_chat(self, parent):
        from tkinter import scrolledtext
        self.chat = scrolledtext.ScrolledText(
            parent,
            font=_FONTS["body"],
            bg=BG, fg=FG_AI,
            insertbackground=ACCENT_1,
            relief=tk.FLAT, bd=0,
            padx=30, pady=22,
            wrap=tk.WORD,
            state=tk.DISABLED,
            cursor="arrow",
            spacing1=2, spacing3=4,
        )
        self.chat.pack(fill=tk.BOTH, expand=True)

        self.chat.vbar.config(
            bg=BG_SIDEBAR, troughcolor=BG,
            width=4, relief=tk.FLAT,
            activebackground="#27272a",
            highlightthickness=0, bd=0,
        )

        t = self.chat
        t.tag_config("sys",         foreground=FG_SYS, font=_FONTS["small"])
        t.tag_config("user_name",   foreground=ACCENT_3, font=_FONTS["label"],
                     spacing1=20, spacing3=5)
        t.tag_config("user_bubble", foreground=FG_USER, font=_FONTS["body"],
                     background=BG_MSG_USER, spacing1=8, spacing3=12)
        t.tag_config("ai_name",     foreground=FG_LABEL, font=_FONTS["label"],
                     spacing1=20, spacing3=5)
        t.tag_config("ai_text",     foreground=FG_AI, font=_FONTS["body"],
                     spacing1=4, spacing3=4)
        t.tag_config("divider",     foreground="#1c1c22", font=_FONTS["small"],
                     spacing1=8, spacing3=8)
        t.tag_config("gap",         foreground=BG, background=BG,
                     font=(_FONTS["body"][0], 2))

    def _build_input(self, parent):
        # Gradient line
        sep = HGradCanvas(parent, c1=ACCENT_1, c2=ACCENT_2, height=1)
        sep.pack(fill=tk.X, side=tk.BOTTOM)

        outer = tk.Frame(parent, bg=BG_INPUT)
        outer.pack(fill=tk.X, side=tk.BOTTOM)

        # Padding frame
        pad = tk.Frame(outer, bg=BG_INPUT, pady=14, padx=20)
        pad.pack(fill=tk.X)

        # Focus border=
        self._border_frame = tk.Frame(pad, bg=BORDER, padx=1, pady=1)
        self._border_frame.pack(fill=tk.X)

        inner = tk.Frame(self._border_frame, bg=BG_INPUT)
        inner.pack(fill=tk.BOTH)

        self.input_box = tk.Text(
            inner,
            font=_FONTS["body"],
            bg=BG_INPUT, fg=FG_USER,
            insertbackground=ACCENT_3,
            relief=tk.FLAT, bd=0,
            padx=14, pady=12,
            height=3, wrap=tk.WORD,
            highlightthickness=0,
        )
        self.input_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.input_box.bind("<Return>",       self._on_enter_key)
        self.input_box.bind("<Shift-Return>", lambda _e: None)
        self.input_box.bind("<FocusIn>",
            lambda _e: self._border_frame.config(bg=BORDER_FOCUS))
        self.input_box.bind("<FocusOut>",
            lambda _e: self._border_frame.config(bg=BORDER))
        self.input_box.focus_set()

        btn_wrap = tk.Frame(inner, bg=BG_INPUT, padx=10)
        btn_wrap.pack(side=tk.RIGHT, fill=tk.Y)
        self.send_btn = SendButton(btn_wrap, command=self._send, bg=BG_INPUT)
        self.send_btn.pack(expand=True)

        # "Thinking" label
        self._think_lbl = tk.Label(pad, text="",
                                   font=_FONTS["small"], fg=FG_LABEL, bg=BG_INPUT,
                                   anchor="w")
        self._think_lbl.pack(fill=tk.X, pady=(6,0))

    # Thinking animation

    def _start_thinking(self):
        self._think_frame = 0
        self._tick_think()

    def _tick_think(self):
        frames = [f"{args.model_name} is thinking ·",
                  f"{args.model_name} is thinking ··",
                  f"{args.model_name} is thinking ···",
                  f"{args.model_name} is thinking ··"]
        self._think_lbl.config(text=frames[self._think_frame % len(frames)])
        self._think_frame += 1
        self._think_job = self.root.after(400, self._tick_think)

    def _stop_thinking(self):
        if self._think_job:
            self.root.after_cancel(self._think_job)
            self._think_job = None
        self._think_lbl.config(text="")

    # Model loading

    def _load_model_async(self):
        threading.Thread(target=lambda: self.root.after(
            0, lambda: self._on_model_loaded(*load_model())), daemon=True).start()

    def _on_model_loaded(self, ok, val_loss, step):
        if ok:
            self.header.set_status("Ready", GREEN)
            self.sidebar.set_status("Ready", GREEN)
            self.sidebar.set_model_info(val_loss, step)
            self._append_sys("Model loaded. Start typing below.\n\n")
        else:
            self.header.set_status("No model", RED)
            self.sidebar.set_status("No model", RED)
            self._append_sys(f"Checkpoint not found: {args.checkpoint}\n")
            self._append_sys("Run  python train.py  first.\n\n")

    # Send / stream

    def _on_enter_key(self, _e):
        self._send()
        return "break"

    def _send(self):
        text = self.input_box.get("1.0", tk.END).strip()
        if not text or not model_loaded or self._generating:
            return
        self._generating = True
        self.input_box.delete("1.0", tk.END)
        self.send_btn.set_state(tk.DISABLED, "···")
        self.header.set_status("Thinking…", AMBER)
        self.sidebar.set_status("Generating…", AMBER)
        self.sidebar.pulse()
        self._start_thinking()
        self._append_user(text)
        self._begin_ai_stream()

        def _run():
            def on_char(ch):
                self.root.after(0, lambda c=ch: self._stream_char(c))
            generate_stream(text, on_char)
            self.root.after(0, self._finish_stream)

        threading.Thread(target=_run, daemon=True).start()

    def _begin_ai_stream(self):
        t = self.chat
        t.config(state=tk.NORMAL)
        t.insert(tk.END, args.model_name + "\n", "ai_name")
        t.mark_set("stream_end", tk.END)
        t.mark_gravity("stream_end", tk.LEFT)
        t.config(state=tk.DISABLED)
        t.see(tk.END)

    def _stream_char(self, ch):
        t = self.chat
        t.config(state=tk.NORMAL)
        t.insert("stream_end", ch, "ai_text")
        t.config(state=tk.DISABLED)
        t.see(tk.END)

    def _finish_stream(self):
        t = self.chat
        t.config(state=tk.NORMAL)
        t.insert(tk.END, "\n\n" + "─" * 58 + "\n", "divider")
        t.config(state=tk.DISABLED)
        t.see(tk.END)
        self._generating = False
        self._stop_thinking()
        self.sidebar.stop_pulse()
        self.header.set_status("Ready", GREEN)
        self.sidebar.set_status("Ready", GREEN)
        self.send_btn.set_state(tk.NORMAL, "Send")
        self.input_box.focus_set()

    # Chat rendering

    def _append_user(self, text):
        t = self.chat
        t.config(state=tk.NORMAL)
        t.insert(tk.END, "YOU\n",       "user_name")
        t.insert(tk.END, text + "\n",   "user_bubble")
        t.insert(tk.END, "\n",          "gap")
        t.config(state=tk.DISABLED)
        t.see(tk.END)

    def _append_sys(self, text):
        t = self.chat
        t.config(state=tk.NORMAL)
        t.insert(tk.END, text, "sys")
        t.config(state=tk.DISABLED)
        t.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.iconbitmap("icon.ico")
    except Exception:
        pass
    app = DarkGPTApp(root)
    root.mainloop()
