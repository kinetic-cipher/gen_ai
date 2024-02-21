from PIL import Image 
from openai import OpenAI
import requests
from io import BytesIO

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
QUERY_TYPE2 = "algebra or calculus or symbolic math query"
QUERY_TYPE3 = "math word problem"
QUERY_TYPE4 = "generic query"
QUERY_TYPE5 = "image generation request"

TEMPERATURE = 0
REQ_TIMEOUT = 20


class MultiModalAgent:

    def __init__(self):

        # base LLM (determine query type)
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature = TEMPERATURE, request_timeout=REQ_TIMEOUT)
        self.prompt_template = PromptTemplate(
            input_variables=["query_type1", "query_type2", "query_type3", "query_type4", "user_input"],
            template="Categorize the following query as either a {query_type1}, {query_type2},"
                     "{query_type3}, {query_type4}, or {query_type5}:\n{user_input}")
        self.llm_chat_category = LLMChain(llm=self.llm, prompt=self.prompt_template)
  
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
        self.palchain = PALChain.from_math_prompt(llm=self.llm, verbose=True, timeout=REQ_TIMEOUT)

        # for image generation requests
        self.open_ai_client = OpenAI()


    def display_image_from_url(self,url, width, height):
        try:
            response = requests.get(url)
            image_data = BytesIO(response.content)
            img = Image.open(image_data)
        
            # Resize the image if necessary
            img = img.resize((width, height))
            img.show()

        except Exception as e:
            print("An error occurred:", e)


    def run(self, user_input: str):

        # query the base llm to categorize the query type
        category_response = self.llm_chat_category.invoke({"query_type1":QUERY_TYPE1,"query_type2":QUERY_TYPE2,
                                                           "query_type3":QUERY_TYPE3, "query_type4":QUERY_TYPE4,
                                                           "query_type5":QUERY_TYPE5,"user_input":user_input})
     
        # call the appropriate chain
        category_response_txt = category_response["text"].lower()
        print(f"\ncategory response: {category_response_txt}")

        if "basic" in category_response_txt:
            print("(interpreting as a basic math query)") 
            response = self.llm_basic_math.invoke(user_input)
            response = response["answer"]
        elif "symbolic" in category_response_txt:
            print("(interpreting as a symbolic math query)") 
            response = self.llm_symbolic_math.invoke(user_input)
            response = response["answer"]
        elif "word" in category_response_txt:
            print("(interpreting as a word problem)") 
            response = self.palchain.invoke(user_input)
            response = response["result"]
        elif "image" in category_response_txt or "generation" in category_response_txt:
            
            # call DALL-E3 for image request
            print("(interpreting as an image generation request)") 
            dalle_response = self.open_ai_client.images.generate(
            model="dall-e-3",
            prompt=user_input,
            size="1024x1024",
            quality="standard",
            n=1,
            )

            # show the result that was pushed in the response url
            url = dalle_response.data[0].url
            self.display_image_from_url(url, 1024,1024)
            response = "image generation complete"
        else:
            print("(interpreting as a generic query)") 
            response = self.llm_chat_generic.invoke({"user_input":user_input})
            response = response["text"]

        return response

