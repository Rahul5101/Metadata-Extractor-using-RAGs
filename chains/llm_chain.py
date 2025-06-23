from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os
load_dotenv()


response_schemas = [
    ResponseSchema(name="agreement_start_date", description="Start date of the rental agreement"),
    ResponseSchema(name="agreement_end_date", description="End date of the rental agreement is 11 months from the start date"),
    ResponseSchema(name="renewal_notice_days", description="Number of days required for renewal notice"),
    ResponseSchema(name="party_one", description="Only Name(s) of the lessor or owner. Do not include the address or s/o"),
    ResponseSchema(name="party_two", description="Only Name(s) of the lessor or owner. Do not include the address or s/o"),
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
few_shot_examples = """
Example 1:
Document: Rental agreement between Prof. A and Mr. B.
Output:
{
  "agreement_start_date": "01/01/2020",
  "agreement_end_date": "01/01/2023",
  "renewal_notice_days": "30",
  "party_one": "Prof. A",
  "party_two": "Mr. B"
}"""
prompt_template = PromptTemplate(
    template="""
Extract the following metadata fields from the rental agreement document text below:

1. Agreement Start Date (07.07.2013 ,print date in this dd.mm.yyyy format)
2. Agreement End Date (add 11 months to the start date)
3. Renewal Notice (is always 30 days)
4. Party One (just write the name of the owner, don't write the s/o or address)
5. Party Two (just write the name of the owner, don't write the s/o or address)

Here is the format of the metadata you have to return:
""This is the agreement between P. JohnsonRavikumar and Saravanan BV starting from 07.07.2013 and ending on 06.06.2014. 
The renewal notice is 30 days. The lessor is P. JohnsonRavikumar and the lessee is Saravanan BV.""



{format_instructions}

Document:
{context}
""",
    input_variables=["context"],
    partial_variables={"format_instructions": output_parser.get_format_instructions(),"examples": few_shot_examples},
)



llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

rag_chain = prompt_template | llm | RunnableLambda(lambda msg: msg.content) | RunnableLambda(output_parser.parse)


def extract_metadata(context_docs):
    context = "\n".join([doc.page_content for doc in context_docs])
    result= rag_chain.invoke({"context": context})
    return result

# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# dummy_context = "This agreement is made between Alice and Bob on 1st January 2020."

# if __name__ == "__main__":
#     result = rag_chain.invoke({"context": dummy_context})
#     print("RAG Chain Output:\n", result)