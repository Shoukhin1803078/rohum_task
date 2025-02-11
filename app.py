import time
import streamlit as st
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
import os
from typing import List, Dict
from langgraph.graph import Graph

st.set_page_config(page_title="Research Assistant Chatbot", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "groq_api_active" not in st.session_state:
    st.session_state.groq_api_active = False
if "tavily_api_active" not in st.session_state:
    st.session_state.tavily_api_active = False
if "workflow" not in st.session_state:
    st.session_state.workflow = None


with st.sidebar:
    st.title("API Configuration")
    
 
    groq_api_key = st.text_input("Enter your GROQ API Key:", type="password")
    if st.button("Activate GROQ API"):
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
            st.session_state.groq_api_active = True
            st.success("GROQ API activated!")
        else:
            st.error("Please enter a GROQ API key")
    
 
    tavily_api_key = st.text_input("Enter your Tavily API Key:", type="password")
    if st.button("Activate Tavily API"):
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
            st.session_state.tavily_api_active = True
            st.success("Tavily API activated!")
        else:
            st.error("Please enter a Tavily API key")
    
    st.write("---")
    st.write("API Status:")
    st.write("GROQ API: ", "🟢 Active" if st.session_state.groq_api_active else "🔴 Inactive")
    st.write("Tavily API: ", "🟢 Active" if st.session_state.tavily_api_active else "🔴 Inactive")


st.title("Research Assistant Chatbot")

def user_raw_query(input_1: str) -> str:
    return input_1

def user_query_optimize_node(input_2: str) -> str:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )
    prompt = f'''Modify this query for internet search to find research-focused results: "{input_2}"
    Return only the optimized search query without any additional text.'''
    response = llm.invoke(prompt).content
    return response

def search_node(input_3: str) -> List[Dict]:
    search = TavilySearchResults(max_results=2, search_depth="advanced")
    results = search.invoke(input_3)
    return results

def final_node(input_4: List[Dict]) -> str:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.5
    )
    sources = "\n".join([f"Source {i+1}: {result['url']}\nContent: {result['content']}\n"
                        for i, result in enumerate(input_4)])
    
    prompt = f'''Generate a detailed research report based on the following sources:
    {sources}
    Format the report as follows:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis
    4. Trends and Insights
    5. Citations
    Ensure all information is properly cited using [Source X] format.'''
    # prompt = f'''Generate a detailed research report based on the following sources:
    # {sources}
    # Format the report as follows:
    # 1. # **Executive Summary**
    # 2. # **Key Findings**
    # 3. # **Detailed Analysis**
    # 4. # **Trends and Insights**
    # 5. # **Citations**
    # Ensure all information is properly cited using [Source X] format.'''
    
    response = llm.invoke(prompt).content
    return response

def initialize_workflow():
    workflow = Graph()
    workflow.add_node("user_raw_query", user_raw_query)
    workflow.add_node("user_query_optimize_node", user_query_optimize_node)
    workflow.add_node("search_node", search_node)
    workflow.add_node("final_node", final_node)
    
    workflow.add_edge("user_raw_query", "user_query_optimize_node")
    workflow.add_edge("user_query_optimize_node", "search_node")
    workflow.add_edge("search_node", "final_node")
    
    workflow.set_entry_point("user_raw_query")
    workflow.set_finish_point("final_node")
    
    return workflow.compile()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask your research question..."):
    if not (st.session_state.groq_api_active and st.session_state.tavily_api_active):
        st.error("Please activate both GROQ and Tavily APIs first!")
    else:
        
        if st.session_state.workflow is None:
            st.session_state.workflow = initialize_workflow()
        
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        
        with st.chat_message("assistant"):
            with st.spinner("Researching..."):
                try:
                    result = st.session_state.workflow.invoke(prompt)
                    print(result)
                    st.markdown(result)
                    # st.write(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    st.error(error_message)





        # Generate response
        # # Generate response
        # with st.chat_message("assistant"):
        #     message_placeholder = st.empty()
        #     with st.spinner("Researching..."):
        #         try:
        #             result = st.session_state.workflow.invoke(prompt)
        #             print(result)
        #             words = result.split()
        #             full_response = ""
        #             for word in words:
        #                 full_response += word + " "
        #                 message_placeholder.markdown(full_response + "▌")
        #                 time.sleep(0.05)  
                    
        #             message_placeholder.markdown(full_response)
        #             st.session_state.messages.append({"role": "assistant", "content": full_response})
        #         except Exception as e:
        #             error_message = f"An error occurred: {str(e)}"
        #             st.error(error_message)



        # # Generate response
        # with st.chat_message("assistant"):
        #     message_placeholder = st.empty()
        #     with st.spinner("Researching..."):
        #         try:
        #             result = st.session_state.workflow.invoke(prompt)
        #             print(result)
                    
        #             
        #             paragraphs = result.split('\n')
        #             full_response = ""
                    
        #             for paragraph in paragraphs:
        #                 full_response += paragraph + "\n"
        #                 
        #                 message_placeholder.markdown(full_response + "▌")
        #                 time.sleep(0.1)  
                    
        #             
        #             message_placeholder.markdown(full_response)
        #             st.session_state.messages.append({"role": "assistant", "content": full_response})
        #         except Exception as e:
        #             error_message = f"An error occurred: {str(e)}"
        #             st.error(error_message)



