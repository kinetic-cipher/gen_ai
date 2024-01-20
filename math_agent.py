from langchain.chains import LLMChain
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_experimental.pal_chain.base import PALChain
from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain

# ignore langchain deprecation warnings
# (remove to see recommended latest APIs to use)
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)


"""
refs:
https://towardsdatascience.com/a-gentle-intro-to-chaining-llms-agents-and-utils-via-langchain-16cd385fca81
https://www.youtube.com/watch?v=DkBc4hfGle8
"""

# constants
QUERY_TYPE1 = "basic math query"
QUERY_TYPE2 = "calculus or symbolic math query"
QUERY_TYPE3 = "math word problem"
QUERY_TYPE4 = "generic query"

TEMPERATURE = 0
REQ_TIMEOUT = 120


class MathAgent:

    def __init__(self):

        # base LLM (determine query type)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = TEMPERATURE, request_timeout=REQ_TIMEOUT)
        self.prompt_template = PromptTemplate(
            input_variables=["query_type1", "query_type2", "query_type3", "query_type4", "user_input"],
            template="Categorize the following query as either a {query_type1}, {query_type2},"
                     "{query_type3}, or {query_type4}:\n{user_input}")
        self.llm_chat = LLMChain(llm=self.llm, prompt=self.prompt_template)
  
        # base LLM (generic)
        self.prompt_template_generic = PromptTemplate(
            input_variables=["user_input"],
            template="{user_input}")
        self.llm_chat_generic = LLMChain(llm=self.llm, prompt=self.prompt_template_generic)


        # basic math LLM
        self.llm_basic_math = LLMMathChain.from_llm(self.llm, verbose=True)

        # symbolic math LLM
        self.llm_symbolic_math=LLMSymbolicMathChain.from_llm(self.llm, verbose=True)

        # word problem LLM
        self.palchain = PALChain.from_math_prompt(llm=self.llm, verbose=True)


    def run(self, user_input: str):

        # query the base llm to categorize the query type
        category_response = self.llm_chat.invoke({"query_type1":QUERY_TYPE1,"query_type2":QUERY_TYPE2,
                                                 "query_type3":QUERY_TYPE3, "query_type4":QUERY_TYPE4, "user_input":user_input})
     
        # run in verbose mode (print full category response)
        print("full category response:")
        print (category_response)


        # call the appropriate chain
        # note: use invoke on generic queries but run on math queries (get errors with invoke)
        category_response_txt = category_response["text"]

        if "basic" in category_response_txt:
            print("\n(interpreting as a basic math query)") 
            response = self.llm_basic_math.run(user_input)
            #response = self.llm_basic_math.invoke(user_input)
            #response = response["answer"]
        elif "symbolic" in category_response_txt:
            print("\n(interpreting as a symbolic math query)") 
            response = self.llm_symbolic_math.run(user_input)
            #response = self.llm_symbolic_math.invoke(user_input)
            #response = response["answer"]
        elif "word" in category_response_txt:
            print("\n(interpreting as a word problem)") 
            response = self.palchain.run(user_input)
            #response = self.palchain.invoke(user_input)
            #response = response["answer"]
        else:
            print("\n(interpreting as a generic query)") 
            response = self.llm_chat_generic.invoke({"user_input":user_input})
            response = response["text"]

        return response


