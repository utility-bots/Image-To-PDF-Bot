import asyncio
import logging
import os
import re
import shutil
import traceback
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque

from PIL import Image, ImageOps
from telethon import TelegramClient, events, errors, Button
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

import database as db
from config import TOKEN, log_channel, API_HASH, API_ID, SESSION, ADMIN


# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
USER_DATA_DIR = BASE_DIR / "UserData"

MAX_IMAGES_PER_USER = 200
MAX_USER_DIR_MB = 250          # total size per user folder
MAX_SINGLE_FILE_MB = 20        # reject huge uploads
USER_DIR_TTL_HOURS = 12        # janitor removes abandoned dirs older than this
JANITOR_INTERVAL_MIN = 30

# PDF size / quality control
PDF_TARGET_DPI = 150
MAX_IMAGE_PIXELS = 6_000_000
JPEG_QUALITY = 80
JPEG_SUBSAMPLING = 1           # 0 best quality, 1 balanced, 2 smallest

# global concurrency controls
MAX_CONCURRENT_DOWNLOADS = 6
MAX_CONCURRENT_CONVERSIONS = 2

DOWNLOAD_SEM = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
CONVERT_SEM = asyncio.Semaphore(MAX_CONCURRENT_CONVERSIONS)


# =========================
# Logging
# =========================
def setup_logger() -> logging.Logger:
    lg = logging.getLogger("pdfbot")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        lg.addHandler(sh)
    return lg


logger = setup_logger()


# =========================
# Robust Telegram call wrapper (FloodWait + Bad Gateway + transient network)
# =========================
def _maybe_error(name: str):
    return getattr(errors, name, None)


# Build a safe transient-exceptions tuple that won't crash if a class doesn't exist
_TRANSIENT_RPC_ERRORS = tuple(
    e for e in [
        _maybe_error("ServerError"),
        _maybe_error("InterdcCallError"),
        _maybe_error("RpcMcgetFailError"),
        _maybe_error("RpcCallFailError"),
        _maybe_error("TimeoutError"),
    ]
    if e is not None
)


async def telethon_call_with_retry(
    coro_factory,
    *,
    max_tries: int = 6,
    base_delay: float = 0.6,
    max_delay: float = 20.0,
):
    """
    Usage:
        await telethon_call_with_retry(lambda: client.send_message(...))
    Retries FloodWait exactly, and transient server/network errors with exponential backoff.
    """
    delay = base_delay
    last_exc = None

    for _attempt in range(1, max_tries + 1):
        try:
            return await coro_factory()

        except errors.FloodWaitError as e:
            await asyncio.sleep(e.seconds + 1)
            continue

        except _TRANSIENT_RPC_ERRORS as e:
            last_exc = e

        except (OSError, TimeoutError, ConnectionError) as e:
            last_exc = e

        except Exception as e:
            # Retry only if it looks transient (502 / bad gateway / timeout)
            last_exc = e
            s = str(e).lower()
            if "bad gateway" not in s and "502" not in s and "timed out" not in s and "timeout" not in s:
                raise

        # backoff + jitter
        jitter = random.uniform(0, 0.25 * delay)
        await asyncio.sleep(min(max_delay, delay + jitter))
        delay = min(max_delay, delay * 2)

    raise last_exc


# =========================
# Helpers: time, dirs, size
# =========================
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def remove_path(p: Path) -> None:
    try:
        if not p.exists():
            return
        if p.is_file():
            p.unlink()
        else:
            shutil.rmtree(p, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to remove {p}: {e}")


def dir_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except Exception:
                pass
    return total


def target_path_from_msg_id(user_dir: Path, msg_id: int) -> Path:
    # idempotent: same telegram message -> same filename
    return user_dir / f"{msg_id}.jpg"


def list_images_sorted(user_dir: Path) -> List[Path]:
    if not user_dir.exists():
        return []

    imgs = [
        p for p in user_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    ]

    def key(p: Path):
        try:
            return (0, int(p.stem))  # message_id.jpg
        except Exception:
            return (1, p.name)

    return sorted(imgs, key=key)


# =========================
# Fix 1) Filename too long + safe join
# =========================
def sanitize_pdf_basename(user_text: str, *, max_len: int = 80) -> str:
    """
    Returns a safe base name WITHOUT extension, capped in length.
    Rejects empty / dots-only.
    """
    s = (user_text or "").strip()

    # remove any extension user tried to add
    s = re.sub(r"\.[A-Za-z0-9]{1,8}$", "", s)

    # keep letters/digits/space/_-.
    s = re.sub(r"[^\w\s\-.]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()

    if not s or s.strip(".") == "":
        return "document"

    s = s[:max_len].rstrip(" .-_")
    return s if s else "document"


def safe_pdf_filename(user_text: str, *, max_len: int = 80) -> str:
    return f"{sanitize_pdf_basename(user_text, max_len=max_len)}.pdf"


def make_internal_pdf_filename(user_id: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{user_id}_{ts}.pdf"


def safe_join_user_file(user_dir: Path, filename: str) -> Path:
    """
    Prevent path traversal; keep file inside user_dir.
    """
    user_dir = user_dir.resolve()
    p = (user_dir / filename).resolve()
    if user_dir not in p.parents and p != user_dir:
        return user_dir / "document.pdf"
    return p


# =========================
# Image normalize + compress (size + orientation)
# =========================
def normalize_and_compress_jpeg_inplace(path: Path) -> None:
    """
    Fix orientation exactly once, convert to RGB, downscale, strip EXIF by not saving it,
    then overwrite as compressed JPEG.
    """
    im = None
    try:
        im = Image.open(path)
        im = ImageOps.exif_transpose(im)

        if im.mode != "RGB":
            im = im.convert("RGB")

        w, h = im.size
        pixels = w * h
        if pixels > MAX_IMAGE_PIXELS:
            scale = (MAX_IMAGE_PIXELS / pixels) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)

        # Do NOT pass exif=... => strips orientation tag
        im.save(
            path,
            format="JPEG",
            quality=JPEG_QUALITY,
            optimize=True,
            progressive=True,
            subsampling=JPEG_SUBSAMPLING,
        )
    finally:
        if im is not None:
            try:
                im.close()
            except Exception:
                pass


# =========================
# Simple rate limiter (anti spam)
# =========================
class RateLimiter:
    def __init__(self):
        self._hits: Dict[int, deque] = {}

    def allow(self, user_id: int, max_events: int = 20, window_sec: int = 10) -> bool:
        dq = self._hits.setdefault(user_id, deque())
        t = now_utc().timestamp()
        dq.append(t)
        cutoff = t - window_sec
        while dq and dq[0] < cutoff:
            dq.popleft()
        return len(dq) <= max_events


# =========================
# Minimal session state
# =========================
@dataclass
class UserState:
    lock: asyncio.Lock
    awaiting_rename: bool = False
    last_seen: datetime = field(default_factory=now_utc)


class StateStore:
    def __init__(self):
        self._states: Dict[int, UserState] = {}

    def get(self, user_id: int) -> UserState:
        st = self._states.get(user_id)
        if st is None:
            st = UserState(lock=asyncio.Lock())
            self._states[user_id] = st
        st.last_seen = now_utc()
        return st


# =========================
# PDF creation (rotation-proof)
# =========================
def image_points_size_fixed_dpi(im: Image.Image, dpi: int = PDF_TARGET_DPI) -> tuple[float, float]:
    w_px, h_px = im.size
    w_pt = w_px * 72.0 / dpi
    h_pt = h_px * 72.0 / dpi
    return w_pt, h_pt


def convert_images_to_pdf_streaming(image_paths: List[Path], pdf_path: Path) -> None:
    """
    Rotation-proof:
    - open each image with PIL
    - exif_transpose again (safe)
    - embed using ImageReader(PIL image)
    """
    ensure_dir(pdf_path.parent)

    c = canvas.Canvas(str(pdf_path))

    for p in image_paths:
        im = None
        try:
            im = Image.open(p)
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                im = im.convert("RGB")

            page_w, page_h = image_points_size_fixed_dpi(im, PDF_TARGET_DPI)
            c.setPageSize((page_w, page_h))

            reader = ImageReader(im)
            c.drawImage(reader, 0, 0, width=page_w, height=page_h)
            c.showPage()
        finally:
            if im is not None:
                try:
                    im.close()
                except Exception:
                    pass

    c.save()


# =========================
# Safe log sender (uses retry wrapper)
# =========================
async def safe_send_log(client: TelegramClient, text: str) -> None:
    try:
        await telethon_call_with_retry(lambda: client.send_message(log_channel, text))
    except Exception:
        pass


def safe_sender_id(event) -> Optional[int]:
    uid = getattr(event, "sender_id", None)
    if isinstance(uid, int) and uid > 0:
        return uid
    return None


# =========================
# Bot
# =========================
class PdfBot:
    def __init__(self):
        ensure_dir(USER_DATA_DIR)
        self.client = TelegramClient(f"{SESSION}tel", API_ID, API_HASH).start(bot_token=TOKEN)
        self.state = StateStore()
        self.rate = RateLimiter()

        self.client.add_event_handler(self.on_start, events.NewMessage(pattern=r"^/start$"))
        self.client.add_event_handler(self.on_help, events.NewMessage(pattern=r"^/help$"))
        self.client.add_event_handler(self.on_cancel, events.NewMessage(pattern=r"^/cancel$"))
        self.client.add_event_handler(self.on_admin, events.NewMessage(pattern=r"^/admin$"))

        self.client.add_event_handler(self.on_media, events.NewMessage(func=self._is_private_image_media))
        self.client.add_event_handler(self.on_album, events.Album(func=lambda e: e.is_private))

        self.client.add_event_handler(self.on_convert, events.CallbackQuery(data=b"convert_to_pdf"))
        self.client.add_event_handler(self.on_remove, events.CallbackQuery(data=b"remove_added_images"))
        self.client.add_event_handler(self.on_new_pdf, events.CallbackQuery(data=b"start_new_pdf"))
        self.client.add_event_handler(self.on_rename_prompt, events.CallbackQuery(data=b"rename_pdf"))

        # IMPORTANT FIX: text-only handler (ignore captions/media)
        self.client.add_event_handler(
            self.on_text_message,
            events.NewMessage(func=self._is_private_text_only),
        )

    def user_dir(self, user_id: int) -> Path:
        return USER_DATA_DIR / str(user_id)

    def main_keyboard(self):
        return [
            [Button.inline("✅ Convert to PDF", b"convert_to_pdf")],
            [Button.inline("🗑 Remove added images", b"remove_added_images")],
            [Button.inline("✏️ Rename output PDF", b"rename_pdf")],
        ]

    def post_keyboard(self):
        return [[Button.inline("➕ Create a new PDF", b"start_new_pdf")]]

    def _is_private_image_media(self, e) -> bool:
        if not e.is_private:
            return False
        if e.photo:
            return True
        if e.document and getattr(e.document, "mime_type", "") and e.document.mime_type.startswith("image/"):
            return True
        return False

    def _is_image_message(self, msg) -> bool:
        if getattr(msg, "photo", None):
            return True
        doc = getattr(msg, "document", None)
        if doc and getattr(doc, "mime_type", "") and doc.mime_type.startswith("image/"):
            return True
        return False

    def _is_private_text_only(self, e) -> bool:
        if not e.is_private:
            return False
        # ignore any media or captions
        if getattr(e, "photo", None):
            return False
        doc = getattr(e, "document", None)
        if doc:
            return False
        txt = getattr(e, "raw_text", None)
        return bool(txt and txt.strip())

    async def track_user(self, event):
        # DB write should never crash bot
        try:
            chat = await event.get_chat()
            chat_id = getattr(chat, "id", None)
            if not isinstance(chat_id, int) or chat_id <= 0:
                return
            username = getattr(chat, "username", None)
            firstname = getattr(chat, "first_name", None)
            lastname = getattr(chat, "last_name", None)
            db.add_user(chat_id, username, firstname, lastname)
        except Exception:
            pass

    async def on_start(self, event):
        await self.track_user(event)
        await telethon_call_with_retry(
            lambda: event.reply(
                "Send images (photos or image files). Then press **Convert to PDF**.\n\n"
                "Commands:\n"
                "/help  /cancel",
                buttons=self.main_keyboard(),
            )
        )

    async def on_help(self, event):
        await telethon_call_with_retry(
            lambda: event.reply(
                "How it works:\n"
                "1) Send images\n"
                "2) Press Convert\n"
                "3) I send the PDF\n\n"
                "Tips:\n"
                "- Use /cancel to wipe your current queue\n"
                "- Use Rename if you want a custom PDF name",
                buttons=self.main_keyboard(),
            )
        )

    async def on_cancel(self, event):
        user_id = safe_sender_id(event)
        if not user_id:
            return

        st = self.state.get(user_id)
        async with st.lock:
            remove_path(self.user_dir(user_id))
            ensure_dir(self.user_dir(user_id))
            st.awaiting_rename = False

        await telethon_call_with_retry(lambda: event.reply("✅ Cleared. Send images again.", buttons=self.main_keyboard()))

    async def on_admin(self, event):
        user_id = safe_sender_id(event)
        if user_id != ADMIN:
            return

        try:
            await telethon_call_with_retry(lambda: self.client.send_file(ADMIN, str(BASE_DIR / "db.sqlite")))
        except Exception:
            pass

        try:
            users = db.get_all_users()
            await telethon_call_with_retry(lambda: self.client.send_message(ADMIN, f"Total Users: {len(list(users))}"))
        except Exception:
            pass

    async def on_media(self, event):
        user_id = safe_sender_id(event)
        if not user_id:
            return

        await self.track_user(event)

        # album parts must be handled by on_album only
        grouped_id = getattr(getattr(event, "message", None), "grouped_id", None)
        if grouped_id:
            return

        if not self.rate.allow(user_id):
            await telethon_call_with_retry(lambda: event.reply("Too many requests. Slow down a bit."))
            return

        st = self.state.get(user_id)
        udir = self.user_dir(user_id)
        ensure_dir(udir)

        try:
            size = getattr(event.file, "size", None)
            if size and size > MAX_SINGLE_FILE_MB * 1024 * 1024:
                await telethon_call_with_retry(lambda: event.reply(f"File too large (>{MAX_SINGLE_FILE_MB}MB)."))
                return
        except Exception:
            pass

        msg_id = getattr(event.message, "id", None)
        if not isinstance(msg_id, int):
            return

        target = target_path_from_msg_id(udir, msg_id)

        async with st.lock:
            if dir_size_bytes(udir) > MAX_USER_DIR_MB * 1024 * 1024:
                await telethon_call_with_retry(lambda: event.reply("Your current queue is too large. Convert or /cancel first."))
                return

            if len(list_images_sorted(udir)) >= MAX_IMAGES_PER_USER:
                await telethon_call_with_retry(lambda: event.reply(f"Too many images (max {MAX_IMAGES_PER_USER}). Convert or /cancel."))
                return

            if target.exists():
                return

        try:
            async with DOWNLOAD_SEM:
                await telethon_call_with_retry(lambda: self.client.download_media(event.message, file=str(target)))

            # CPU work outside semaphore
            try:
                await asyncio.to_thread(normalize_and_compress_jpeg_inplace, target)
            except Exception as e:
                logger.warning(f"compress failed for {target}: {e}")

            count = len(list_images_sorted(udir))
            await telethon_call_with_retry(lambda: event.reply(f"✅ Added. Total images: {count}", buttons=self.main_keyboard()))

        except Exception as e:
            err = f"Download error: {e}\n\n{traceback.format_exc()}"
            logger.error(err)
            await safe_send_log(self.client, err)

    async def on_album(self, event):
        user_id = safe_sender_id(event)
        if not user_id:
            return

        await self.track_user(event)

        st = self.state.get(user_id)
        udir = self.user_dir(user_id)
        ensure_dir(udir)

        msgs = [m for m in event.messages if self._is_image_message(m)]
        if not msgs:
            return

        msgs.sort(key=lambda m: m.id)

        async with st.lock:
            if dir_size_bytes(udir) > MAX_USER_DIR_MB * 1024 * 1024:
                await telethon_call_with_retry(lambda: self.client.send_message(user_id, "Your current queue is too large. Convert or /cancel first."))
                return

            existing_count = len(list_images_sorted(udir))
            if existing_count + len(msgs) > MAX_IMAGES_PER_USER:
                await telethon_call_with_retry(lambda: self.client.send_message(user_id, f"Too many images. Max {MAX_IMAGES_PER_USER}. Convert or /cancel."))
                return

        downloaded_any = False
        targets: List[Path] = []

        try:
            async with DOWNLOAD_SEM:
                for msg in msgs:
                    mid = getattr(msg, "id", None)
                    if not isinstance(mid, int):
                        continue
                    target = target_path_from_msg_id(udir, mid)
                    if target.exists():
                        continue

                    await telethon_call_with_retry(lambda m=msg, t=target: self.client.download_media(m, file=str(t)))
                    targets.append(target)
                    downloaded_any = True

            # compress outside semaphore
            for t in targets:
                try:
                    await asyncio.to_thread(normalize_and_compress_jpeg_inplace, t)
                except Exception as e:
                    logger.warning(f"compress failed for {t}: {e}")

            if downloaded_any:
                count = len(list_images_sorted(udir))
                await telethon_call_with_retry(
                    lambda: self.client.send_message(user_id, f"✅ Album added. Total images: {count}", buttons=self.main_keyboard())
                )

        except Exception as e:
            err = f"Album error: {e}\n\n{traceback.format_exc()}"
            logger.error(err)
            await safe_send_log(self.client, err)

    async def on_remove(self, event):
        user_id = safe_sender_id(event)
        if not user_id:
            return

        st = self.state.get(user_id)
        async with st.lock:
            remove_path(self.user_dir(user_id))
            ensure_dir(self.user_dir(user_id))
            st.awaiting_rename = False

        try:
            await event.answer("Deleted.")
        except Exception:
            pass

        await telethon_call_with_retry(lambda: self.client.send_message(user_id, "🗑 Removed. Send images again.", buttons=self.main_keyboard()))

    async def on_new_pdf(self, event):
        user_id = safe_sender_id(event)
        if not user_id:
            return

        st = self.state.get(user_id)
        async with st.lock:
            remove_path(self.user_dir(user_id))
            ensure_dir(self.user_dir(user_id))
            st.awaiting_rename = False

        try:
            await event.answer()
        except Exception:
            pass

        await telethon_call_with_retry(lambda: self.client.send_message(user_id, "Send images for the new PDF.", buttons=self.main_keyboard()))

    async def on_rename_prompt(self, event):
        user_id = safe_sender_id(event)
        if not user_id:
            return

        st = self.state.get(user_id)
        async with st.lock:
            st.awaiting_rename = True

        try:
            await event.answer()
        except Exception:
            pass

        await telethon_call_with_retry(lambda: self.client.send_message(user_id, "Send the PDF name (no .pdf). Example: my_notes"))

    async def on_text_message(self, event):
        # Only handle rename input; ignore commands
        if not event.raw_text or event.raw_text.startswith("/"):
            return

        user_id = safe_sender_id(event)
        if not user_id:
            return

        st = self.state.get(user_id)

        async with st.lock:
            if not st.awaiting_rename:
                return

            st.awaiting_rename = False

            udir = self.user_dir(user_id)
            images = list_images_sorted(udir)
            if not images:
                await telethon_call_with_retry(lambda: event.reply("No images queued. Send images first.", buttons=self.main_keyboard()))
                return

            raw = event.raw_text.strip()

            # Reject empty/just dots
            if not raw or raw.strip(".") == "":
                await telethon_call_with_retry(lambda: event.reply("Invalid name. Send something like: my_notes", buttons=self.main_keyboard()))
                return

            pdf_name = safe_pdf_filename(raw, max_len=80)

            if pdf_name.lower() == ".pdf" or pdf_name.strip(".").lower() == "pdf":
                await telethon_call_with_retry(lambda: event.reply("Invalid name. Send something like: my_notes", buttons=self.main_keyboard()))
                return

            pdf_path = safe_join_user_file(udir, pdf_name)

        wait = await telethon_call_with_retry(lambda: event.reply("⏳ Creating PDF..."))
        ok = await self._convert_and_send(user_id, pdf_path, wait_message_id=getattr(wait, "id", None))
        if not ok:
            await telethon_call_with_retry(lambda: self.client.send_message(user_id, "❌ Failed. Try again.", buttons=self.main_keyboard()))

    async def on_convert(self, event):
        user_id = safe_sender_id(event)
        if not user_id:
            return

        udir = self.user_dir(user_id)
        images = list_images_sorted(udir)
        if not images:
            try:
                await event.answer("No images.")
            except Exception:
                pass
            await telethon_call_with_retry(lambda: self.client.send_message(user_id, "Send images first.", buttons=self.main_keyboard()))
            return

        try:
            await event.answer()
        except Exception:
            pass

        pdf_path = safe_join_user_file(udir, make_internal_pdf_filename(user_id))
        status = await telethon_call_with_retry(lambda: self.client.send_message(user_id, "⏳ Processing..."))

        ok = await self._convert_and_send(user_id, pdf_path, wait_message_id=getattr(status, "id", None))
        if not ok:
            await telethon_call_with_retry(lambda: self.client.send_message(user_id, "❌ Failed. Try again.", buttons=self.main_keyboard()))

    async def _convert_and_send(self, user_id: int, pdf_path: Path, wait_message_id: Optional[int] = None) -> bool:
        st = self.state.get(user_id)
        udir = self.user_dir(user_id)

        success = False

        async with CONVERT_SEM:
            try:
                async with st.lock:
                    images = list_images_sorted(udir)
                    if not images:
                        return False

                # IMPORTANT FIX: run heavy PDF conversion off the event loop
                await asyncio.to_thread(convert_images_to_pdf_streaming, images, pdf_path)

                if not pdf_path.exists():
                    return False

                await telethon_call_with_retry(lambda: self.client.send_file(user_id, str(pdf_path), buttons=self.post_keyboard()))
                success = True

                if wait_message_id is not None:
                    try:
                        await telethon_call_with_retry(lambda: self.client.delete_messages(user_id, wait_message_id))
                    except Exception:
                        pass

                return True

            except Exception as e:
                err = f"Conversion/send error: {e}\n\n{traceback.format_exc()}"
                logger.error(err)
                await safe_send_log(self.client, err)
                return False

            finally:
                # Cleanup strategy:
                # - If success: wipe everything immediately
                # - If failure: delete only the pdf (keep images for retry; janitor will remove later)
                try:
                    if success:
                        async with st.lock:
                            remove_path(udir)
                            ensure_dir(udir)
                    else:
                        remove_path(pdf_path)
                except Exception:
                    pass

    async def janitor(self):
        while True:
            try:
                cutoff = now_utc() - timedelta(hours=USER_DIR_TTL_HOURS)
                if USER_DATA_DIR.exists():
                    for d in USER_DATA_DIR.iterdir():
                        if not d.is_dir():
                            continue
                        try:
                            # IMPORTANT FIX: timezone-aware mtime
                            mtime = datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)
                        except Exception:
                            continue
                        if mtime < cutoff:
                            remove_path(d)
                            logger.info(f"[janitor] removed old dir: {d}")
            except Exception as e:
                logger.warning(f"[janitor] error: {e}")
            await asyncio.sleep(JANITOR_INTERVAL_MIN * 60)

    def run(self):
        self.client.loop.create_task(self.janitor())
        logger.info("PDF bot started.")
        self.client.run_until_disconnected()


if __name__ == "__main__":
    PdfBot().run()
