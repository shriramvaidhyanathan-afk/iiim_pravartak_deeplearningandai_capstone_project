import os
import json
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path


class TraceabilityManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TraceabilityManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_dir: str = "storage"):
        # Prevent re-initialization on every call
        if self._initialized:
            return

        self.base_dir = Path(base_dir)
        self._initialize_session()
        self._initialized = True

    def add_internal_prompt_response(self, prompt, response):
        self._internal_prompts_responses.append({"prompt": prompt, "response": response})

    def _initialize_session(self):
        """Internal helper to set up a new session state."""

        self.conversation_id = f"conv_{TraceabilityManager._get_current_time()}_{uuid.uuid4().hex[:8]}"
        self.conv_path = self.base_dir / self.conversation_id
        self.doc_path = self.conv_path / "documents"
        self.doc_images_path = self.doc_path / "images"
        self.doc_images_text_path = self.doc_images_path / "text"
        self.doc_text_path = self.doc_path / "text"
        self.logs_path = self.conv_path / "logs"
        self.prompts_path = self.conv_path / "prompts"
        self.model_name = None
        self._internal_prompts_responses = []
        self.user_feedback = ""

        # Initialize folder structure
        self._setup_folders()

        # initialize logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        c_handler = logging.StreamHandler()  # Console
        f_handler = logging.FileHandler(self.logs_path / f'traceability___{TraceabilityManager._get_current_time()}.log')
        c_handler.setLevel(logging.WARNING)  # Only show warnings+ in console
        f_handler.setLevel(logging.INFO)  # Save info+ in file

        # 3. Create formatters and add them to handlers
        format_str = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(format_str)
        f_handler.setFormatter(format_str)

        # 4. Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        # In-memory record
        self.history = {
            "conversation_id": self.conversation_id,
            "start_time": datetime.now().isoformat(),
            "interactions": {}
        }
        logging.info(f"Traceability Session Initialized: {self.conversation_id}")

    def hard_reset(self):
        """
        Explicitly clears the current session and starts a brand new one.
        Useful when a user starts a fresh chat or uploads a new batch of files.
        """
        logging.warning("Performing Hard Reset on Traceability Manager...")
        self._initialize_session()

    def _setup_folders(self):
        """Creates the physical storage hierarchy on your SSD."""
        paths = [
            self.conv_path,
            self.doc_path,
            self.doc_text_path,
            self.doc_images_path,
            self.logs_path,
            self.prompts_path
        ]
        for p in paths:
            p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_current_time():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_original_redacted_image(self, original_img, redacted_img, file_prefix, page_number, image_index):
        current_datetime = TraceabilityManager._get_current_time()
        original_file_path = self.doc_images_path / f"{file_prefix}___{page_number}___{image_index}___original.png"
        redacted_file_path = self.doc_images_path / f"{file_prefix}___{page_number}___{image_index}___redacted.png"
        with open(original_file_path, "wb") as f:
            f.write(original_img)
        with open(redacted_file_path, "wb") as f:
            f.write(redacted_img)

    def add_original_redacted_text(self, original: str, redacted: str, file_prefix, page_number, image_index=None,
                                   is_image_text=False):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "original": original,
            "redacted": redacted
        }

        if not is_image_text:
            file_name = f"{file_prefix}___{page_number}.json"
            with open(self.doc_text_path / file_name, "w") as f:
                json.dump(entry, f, indent=4)
        else:
            file_name = f"{file_prefix}___{page_number}___{image_index}.json"
            with open(self.doc_text_path / file_name, "w") as f:
                json.dump(entry, f, indent=4)

    def track_file(self, original_path: str):
        """Copies the uploaded PDF into the traceability folder."""
        file_name = os.path.basename(original_path)
        dest = self.doc_path / file_name
        shutil.copy(original_path, dest)
        return str(dest)

    def track_interaction(self, interaction_id, user_input: str, user_redacted_input: str, assistant_output: str):
        """Records a single Q&A turn."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "user_prompt": user_input,
            "user_redacted_prompt": user_redacted_input,
            "internal_prompts_responses": self._internal_prompts_responses,
            "response": assistant_output,
            "user_feedback": ""
        }
        self.history["interactions"][interaction_id] = entry

    def track_user_feedback(self, interaction_id, user_feedback: str):
        self.history["interactions"][interaction_id]["user_feedback"] = user_feedback

    def get_conversation_id(self):
        return self.conversation_id

    def save_metadata(self):
        with open(self.prompts_path / f"interactionmetadata.json", "w") as f:
            json.dump(self.history, f, indent=4)


trace_manager = TraceabilityManager()

