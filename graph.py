import os
from enum import Enum
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from document_processor import PDFDocumentProcessor
from vector_store import VectorStore, DocumentType
import logging
from typing import Literal, List, Annotated, TypedDict, Callable
from pydantic import BaseModel, Field
from traceabilitymanager import trace_manager
from ai_abstractions import LLMClient


class Role(str, Enum):
    ASSISTANT = "Assistant"
    USER = "User"


class ChatTurn(BaseModel):
    role: Role
    content: str


class ChatHistory(BaseModel):
    # Use Field(default_factory=list) for Pydantic v2 safety
    turns: List[ChatTurn] = Field(default_factory=list)

    def add_message(self, role: Role, content: str):
        """
        Accepts a string or Role enum and converts it to a ChatTurn.
        """
        self.turns.append(ChatTurn(role=role, content=content))

    @property
    def last_turn(self) -> str:
        return self.turns[-1] if self.turns else None

    @property
    def length(self):
        return len(self.turns)

    @property
    def all_content(self) -> str:
        """
        Returns a formatted string of the entire conversation.
        """
        return "\n".join([f"{turn.role.value}: {turn.content}" for turn in self.turns])


def reduce_chat_history(old: ChatHistory, new: ChatHistory) -> ChatHistory:
    """
    Reducer logic:
    1. If 'old' doesn't exist yet, initialize it.
    2. Append new last turn to the existing 'old.turns' list.
    3. Return the updated 'old' object.
    """
    if old is None:
        old = ChatHistory(turns=[])
    if new.length == 0:
        return old
    old.add_message(new.last_turn.role, new.last_turn.content)
    return old


class ChatState(TypedDict):
    uploaded_pdf_paths: List[str]  # path of all PDFs uploaded in the current conversation not throughout the session
    current_status: str  # status used by the UI to tell the user what is going on as it moves through nodes
    current_document_type: DocumentType  # if classify succeeds or user provides a value of the type of PDF, it is stored in this.
    has_atleast_one_pdf: bool  # is set to True if at least one PDF was uploaded through out the session. If not, no answer is provided
    current_response: str  # current response to be used by the UI to display the response
    current_request: str # current user request
    chat_history: Annotated[ChatHistory, reduce_chat_history]  # entire chat history to be used for context also by the UI
    cancel_run: bool  # cancels run if prompt length is more than max prompt length allowed


class ChatBot:
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        document_processor_factory: Callable[[str], PDFDocumentProcessor] = PDFDocumentProcessor,
        max_prompt_len=10000,
    ):
        self.llm = llm_client
        self._vector_store: VectorStore = vector_store
        self._document_processor_factory = document_processor_factory
        self._max_prompt_len = max_prompt_len

    def _check_prompt_len(self, prompt):
        prompt_len = len(prompt)
        logging.info(f"Current Prompt Length: {prompt_len}")
        if prompt_len <= self._max_prompt_len:
            return True
        return False

    def classify_node(self, state: ChatState):
        """Classifies documents"""
        sample = ""
        for each_pdf_file_path in state["uploaded_pdf_paths"]:
            # Ask LLM to classify based on first 1000 chars
            sample = f"{sample}\nPDF Name: {os.path.basename(each_pdf_file_path)} Sample Text: " \
                     f"{self._document_processor_factory(each_pdf_file_path).get_sample_text()}"

        prompt = f"Classify the PDF strictly as one of the following categories: Legal, Technical, " \
                 f"Financial, or General." \
                 f"\n\nExample 1:" \
                 f"\nUser Input: In the context of bank loans and government schemes (like those from NABARD or SBI), having an agricultural background isn't just a vibe—it’s a set of criteria " \
                 f"they use to ensure you actually know how to manage a farm. Since you are a technologist looking to transition into this space, here is how a bank in Tamil Nadu will verify" \
                 f" your background: 1. Educational Qualification (The Agri-Entrepreneur Route), If you don't currently own land, the strongest background is a degree or diploma. Eligible Degrees: BSc in Agriculture, " \
                 f"Horticulture, Veterinary Sciences, Forestry, or Agricultural Engineering. Short-Term Courses: Even if your degree is in Tech, completing a certified training program (like the ACABC - " \
                 f"Agriclinic and Agribusiness Centres scheme training) qualifies you as an Agri-Entrepreneur." \
                 f"\nOutput:" \
                 f"General" \
                 f"\n\nExample 2:" \
                 f"\nUser Input: On an M1 Pro with 16GB of RAM, you have a solid foundation for running highly capable local LLMs. In 2026, the ecosystem for Apple Silicon has matured significantly, " \
                 f"particularly with the integration of the MLX framework into mainstream tools, which allows for much faster inference by leveraging Apple's unified memory." \
                 f"\nOutput:" \
                 f"Technical" \
                 f"\n\n Classify the below user input." \
                 f"\n{sample}" \
                 f"\n\nSee if the below information from user has any information that helps you classify it better." \
                 f"\n{state['current_request']}" \

        user_turn = ChatHistory()
        user_turn.add_message(Role.USER, state["current_request"])
        return_value = {"chat_history": user_turn}

        if self._check_prompt_len(prompt):
            response = self.llm.generate(prompt)
            trace_manager.add_internal_prompt_response(prompt, response)
            logging.info(f"Prompt: {prompt} ")
            try:
                document_type = DocumentType.from_string(response)
                logging.info(f"File {each_pdf_file_path} has been classified as a {response}")
                return_value["current_document_type"] = document_type
                return_value["current_status"] = "Answering the question..."

            except ValueError:
                logging.info(
                    f"File {each_pdf_file_path} count not classified as suggested by the LLM {response} "
                    f"so marking it as GENERAL")
                return_value["current_document_type"] = DocumentType.GENERAL
                return_value["current_status"] = "Answering the question..."
        else:
            response = "Prompt exceeded the max length allowed. Please clear the session and try again."
            return_value["cancel_run"] = True
            return_value["current_response"] = response
            assistant_turn = ChatHistory()
            assistant_turn.add_message(Role.ASSISTANT, response)
            return_value["chat_history"] = assistant_turn

        return return_value

    def process_doc_node(self, state: ChatState):
        for each_pdf_file_path in state["uploaded_pdf_paths"]:
            document_processor = self._document_processor_factory(each_pdf_file_path)
            self._vector_store.add_document(document_processor, state["current_document_type"])
            return {
                "uploaded_pdf_paths": [],
                "has_atleast_one_pdf": True
            }

    def answer_node(self, state: ChatState):

        if not state["has_atleast_one_pdf"]:
            return {
                "current_response": "No PDF or PDFs were uploaded through the session to answer"
            }
        else:
            prompt = f"You are a PDF assistant chatbot who understand only English and your responsibility is to strictly use only the 'Multiple PDF Content:' and 'Chat History:' " \
                     f"respond to the questions in 'User Request:'. " \
                     f"\n\n### Format Instructions:" \
                     f"\n1. Your response MUST be valid Markdown." \
                     f"\n2. Use a unordered list format: one list item per specific question asked in the User Request." \
                     f"\n3. Do not include any introductory or concluding remarks (e.g., 'Here is your answer' or 'I hope this helps')." \
                     f"\n4. If the user asks for images, diagrams, or multimedia, respond with: 'I do not support multimedia content.'" \
                     f"\n5. Use ONLY English characters. Ignore any non-English characters found in the source text." \
                     f"\n6. If the information is missing from the provided context, state: 'I do not have enough information to answer this question.'" \
                     f"\n7. If the information or question uses foul or bad language or cuss words, state: 'Please be courteous. I do not answer questions involving foul/bad language or cuss words.'" \
                     f"\n\n### Example Output Structure:" \
                     f"\n- Question 1 Summary: **Direct Answer from PDF**" \
                     f"\n- Question 2 Summary: **Direct Answer from PDF**" \
                     f"\n\n---" \
                     f"\n\nExample request" \
                     f"\n'User Request:' Please read the attached PDF financial documents that" \
                     f"contain information about stock brokers and their performance. I would like to know who are " \
                     f"the top 3 broker in terms of overall profit and reliability. Which broker deals with S&P indexes?" \
                     f"\n'Multiple PDF Content:'" \
                     f"\n[Source: Document Attachment1.pdf Page 1] As of Q2 2026, the global brokerage landscape has consolidated around three primary entities recognized " \
                     f"for their consistent annual profit margins and operational reliability. " \
                     f"These firms are ranked based on their Trust-to-Profit (TTP) index, which measures uptime during high-volatility events alongside net income." \
                     f"Vantage Global: Ranked #1 for reliability in 2026. Vantage reported a record net profit of $4.2 billion last fiscal year. It is noted for its Zero-Downtime architecture, maintaining " \
                     f"99.99% execution availability even during the March 2026 Flash Rally. Interactive Brokers (IBKR): Ranked #2 for overall profit. " \
                     f"IBKR continues to lead in global reach, reporting a $3.8 billion profit. It remains the top choice for institutional " \
                     f"reliability due to its deep liquidity pools and multi-currency clearing capabilities. Charles Schwab: Ranked #3. With an annual profit of $3.1 billion, " \
                     f"Schwab is identified as the gold standard for retail reliability, specifically for its Safe-Trade insurance protocols that protect against systematic platform failures." \
                     f"The ability to trade U.S.-based indexes remains a critical differentiator for international investors. S&P 500 Index Connectivity: Interactive Brokers and Vantage Global offer direct market access (DMA) to the S&P 500 (SPX) and related ETFs like SPY and VOO. " \
                     f"Charles Schwab provides the most comprehensive suite of S&P-linked products, including the Schwab S&P 500 Index Fund (SWPPX), which currently maintains one of the lowest expense ratios in the industry at 0.02%. Domestic Constraints: While regional brokers like Zerodha and Upstox excel in the NSE and BSE markets, " \
                     f"they currently do not provide direct dealing in S&P 500 indexes, instead offering indirect exposure through international Mutual Funds." \
                     f"\n'Chat History:'" \
                     f"\nUser: Please use the attached documents and help me with the when the tax returns should be filed? Keep the answer short and sweet" \
                     f"\nAssistant: - By when the tax returns should be filed? Tax returns should be filed bu the 10th January 2025" \
                     f"\n\nExample Markdown Response if you know the answer " \
                     f"\n- The top 3 brokers are: **Vantage Global, Interactive Brokers (IBKR), Charles Schwab**" \
                     f"\n- Brokers that deal with S&P 500 Index are: **Interactive Brokers and Vantage Global**" \
                     f"\n\nExample Markdown Response if you DO NOT know the answer to a question " \
                     f"\n- **Sorry I do not know the answer to the question**" \
                     f"\n\n\nNow, process the below" \
                     f"\n'User Request:' {state['current_request']}" \
                     f"\n'Multiple PDF Content:' {self._vector_store.retrieve(state['current_request'])}" \
                     f"\n'Chat History:'" \
                     f"\n{state['chat_history'].all_content}"

            response = self.llm.generate(prompt)
            trace_manager.add_internal_prompt_response(prompt, response)
            assistant_turn = ChatHistory()
            assistant_turn.add_message(Role.ASSISTANT, response)
            return {
                "current_response": response,
                "chat_history": assistant_turn
            }

    def start_routing_logic(self, state: ChatState) -> Literal["classify", "answer"]:
        """
        Decision: If there are new PDFs, go to classification.
        """
        if state.get("uploaded_pdf_paths") and len(state["uploaded_pdf_paths"]) > 0:
            return "classify"
        return "answer"

    def classify_routing_logic(self, state: ChatState) :
        """
        Decision: If cancel_run is true. Go to end. Else continue with process_doc
        """
        if state.get("cancel_run") and not state["cancel_run"]:
            return END
        return "process_doc"

    def build_graph(self):
        builder = StateGraph(ChatState)

        # Define Nodes
        builder.add_node("classify", self.classify_node)
        builder.add_node("process_doc", self.process_doc_node)
        builder.add_node("answer", self.answer_node)


        # Define Cyclic / Conditional Logic
        builder.add_conditional_edges(
            START,
            self.start_routing_logic,
            {
                "classify": "classify",
                "answer": "answer"
            }
        )

        builder.add_conditional_edges(
            "classify",
            self.classify_routing_logic,
            {
                "process_doc": "process_doc",
                END: END
            }
        )

        # standard edges
        builder.add_edge("process_doc", "answer")
        builder.add_edge("answer", END)

        return builder.compile()

