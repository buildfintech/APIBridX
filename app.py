import streamlit as st
import requests
import pandas as pd
import threading
from fastapi import FastAPI, Request, HTTPException
import uvicorn
import jwt
import datetime
import requests
import platform
import threading
import random
import string
import multiprocessing
import traceback
from multiprocessing import Queue
import io
import sys
import contextlib
import io
import sys
import contextlib
import multiprocessing
import traceback
import json
import ast

def _maskAPIKey(key):
    return '*' * (4) + key[-4:] if len(key)> 4 else '*'

unique_thread_name = "_fastapi_proxy_thread_88765"

# Function to generate a unique session ID
def generate_session_id():
     return ''.join(random.choices(string.ascii_letters + string.digits, k=15))

def find_fastapi_server_thread(name=unique_thread_name):
    for thread in threading.enumerate():
        if thread.name == name:
            return thread
    return None

class JWTKeyStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.jwt_keys = {}
                cls._instance.JWT_SECRET = '_Thentral_NAM8Bker#V*)'
                cls._instance.JWT_ALGORITHM = 'HS256'
        return cls._instance

    def create_jwt_token(self, session_id, service_name, openai_key, hours_valid=24):
        expiry_time = datetime.datetime.utcnow() + datetime.timedelta(hours=hours_valid)
        payload = {"service_name": service_name, "exp": expiry_time}
        token = jwt.encode(payload, self.JWT_SECRET, algorithm=self.JWT_ALGORITHM)

        with self._lock:
            if session_id not in self.jwt_keys:
                self.jwt_keys[session_id] = {}
            self.jwt_keys[session_id][(service_name, token)] = openai_key
        return token, expiry_time

    def get_service_key(self, session_id, service_name, token):
        with self._lock:
            return self.jwt_keys.get(session_id, {}).get((service_name, token))

    def is_valid_session(self, session_id):
        with self._lock:
            return session_id in self.jwt_keys

    def decode_jwt(self, token):
        with self._lock:
            return jwt.decode(token, self.JWT_SECRET, algorithms=[self.JWT_ALGORITHM])




class FastAPIProxy:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.initialized = False
        return cls._instance

    @classmethod
    def instance(cls):
        inst = cls.__new__(cls)
        if not inst.initialized:
            inst.__init__()
            inst.initialized = True
        return inst

    def __init__(self):
        if not self.initialized:
            self.app = FastAPI()
            self.logs = {}
            self.jwt_key_store_ref = JWTKeyStore()
            self.config = uvicorn.Config(self.app)
            self.config_host = self.config.host
            self.config_port = self.config.port
            self.setup_routes()

    def setup_routes(self):

        @self.app.get("/")
        async def read_root():
            current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            return {"systemtime": current_time, "message": "This is a Proxy Server for API keys", }

        @self.app.get("/logs")
        async def get_logs(session_id: str):
            if session_id in self.logs:
                return self.logs[session_id]
            else:
                raise HTTPException(status_code=401, detail= "Not a validate session")
            
        @self.app.post("/proxy")
        async def proxy_request(request: Request):
            session_id   = request.headers.get('Session_Id')

            if not self.jwt_key_store_ref.is_valid_session(session_id):
                raise HTTPException(status_code=401, detail= "Missing Session ID")
            
            
            if session_id not in self.logs:
                self.logs[session_id] = []  

            # Initial log data
            log_data = {
                "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id" : session_id,
                "service_name": None,
                "original_url": None,
                "status": "Failed",
                "detail": None,
                "api_key" :None
            }

            token = request.headers.get('Authorization')
            service_name = request.headers.get('Service-Name')
            
            log_data["service_name"] = service_name

            if not token or not service_name:
                log_data["detail"] = "Missing token or Service-Name"
                self.logs[session_id].append(log_data)
                raise HTTPException(status_code=401, detail=log_data["detail"])

            try:
                # Decode JWT token using JWTKeyStore instance
                decoded_token = self.jwt_key_store_ref.decode_jwt(token)
                # Retrieve OpenAI API key for the given token
                api_key = self.jwt_key_store_ref.get_service_key(session_id, service_name,token)
                if not api_key:
                    log_data["detail"] = "OpenAI API key not found for token"
                    self.logs[session_id].append(log_data)
                    raise HTTPException(status_code=400, detail=log_data["detail"])
            except jwt.ExpiredSignatureError:
                log_data["detail"] = "Expired token"
                self.logs[session_id].append(log_data)
                raise HTTPException(status_code=401, detail=log_data["detail"])
            except jwt.InvalidTokenError:
                log_data["detail"] = "Invalid token"
                self.logs[session_id].append(log_data)
                raise HTTPException(status_code=401, detail=log_data["detail"])

            original_request_data = await request.json()
            log_data["original_url"] = original_request_data.get('original_url')

            try:
                response = requests.post(
                    url=original_request_data['original_url'],
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=original_request_data.get("original_body")
                )
                log_data["status"] = "Processed"
                log_data["detail"] = "Request processed successfully"
                log_data["api_key"] = _maskAPIKey(api_key)
            except Exception as e:
                log_data["detail"] = f"Request processing failed: {str(e)}"
                raise HTTPException(status_code=500, detail=log_data["detail"])


            self.logs[session_id].append(log_data)  
            return response.json()
        
            
    def run(self):
        try:
            # config = uvicorn.Config(self.app, host=self.host, port=8000)
            
            server = uvicorn.Server(self.config)
            server.run()
        except Exception as e:
            SystemError("FastAPI server failed to start")

def load_log_data():
    # URL of the FastAPI logs endpoint
    hostname = FastAPIProxy.instance().config_host
    port = FastAPIProxy.instance().config_port
    session_id = st.session_state['session_id']
    url = "http://{host}:{port}/logs".format(host=hostname, port=port)
    response = requests.get(url,  params={'session_id': session_id})
    if response.status_code == 200:
        st.session_state['history_data_loaded'] = True
        return response.json()
    

# Define the function to run user code at the module level
def run_user_code(user_code, output_queue, error_queue):
    try:
        # Redirect standard output
        with contextlib.redirect_stdout(io.StringIO()) as output_buffer:
            exec(user_code)

        # Get the output and put it in the output queue
        output = output_buffer.getvalue()
        output_queue.put(output)
    except Exception:
        # Capture the exception and put it in the error queue
        error_info = traceback.format_exc()
        error_queue.put(error_info)


def execute_code_in_process(user_code):
    output_queue = multiprocessing.Queue()
    error_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=run_user_code, args=(user_code, output_queue, error_queue))
    process.start()
    return process, output_queue, error_queue


def show_code(sessionid, token, proxy_url, msg):
    code_template = '''
#1. Form a request
request_data = {{
    "original_url": "https://api.openai.com/v1/chat/completions",
    "original_body": {{
        "model": "gpt-3.5-turbo-1106",  # Specify the model
        "messages": [{{"role": "user", "content": "{msg}"}}]
    }}
}}
#2. API call routed via Proxy server
response = requests.post(
    "{proxy_url}/proxy",
    headers={{
        "Authorization": "{token}",
        "Service-Name": "openai",
        "Session_Id": "{sessionid}"
    }},
    json=request_data 
)
print(json.loads(response.text))
    '''

    formatted_code = code_template.format(token=token, sessionid=sessionid, proxy_url=proxy_url, msg=msg)
    st.code(formatted_code, language='python')

    return formatted_code

    
def show_image():
    st.image('TokenFlow.jpg', use_column_width=True)

def on_button_register_click():
    st.session_state['btn_action_register'] = True

def on_button_revoke_click():
    st.session_state['btn_action_revoke'] = True

def setupsidebar_registration(proxy_url):
    with st.sidebar:
        # FastAPIProxy running status
        with st.expander("Proxy Server"):
            proxy_status = "Running" if find_fastapi_server_thread()  else "Stopped"
            color = ":green" if proxy_status == "Running" else ":red"
            st.write(f"Status: {color}[{proxy_status}]")
            st.write(f"Session ID: :blue[{st.session_state['session_id']}]")
            st.write(f"Proxy : {proxy_url}")

        st.title("1.Generate Token")

        # Service Name Selection - Single Select
        service_name_reg = st.selectbox("Select API Key *", ["openai"], key="service_reg")

        # Conditional Input for OpenAI Key
        openai_key = st.text_input("OpenAI API Key *", key="openai_key", type="password")

        # Session Expiry
        expiry_hours_reg = st.number_input("Token Expiry (in hours) *", min_value=1, max_value=24, value=5, key="expiry_hours_reg")

        # Generate Button - Only active if all fields are provided
        all_fields_provided = all([service_name_reg, openai_key, expiry_hours_reg > 0])
        gen_button = st.button("Register", key="register_session", on_click=on_button_register_click, disabled=not all_fields_provided)

    if st.session_state.get('btn_action_register', False) and all_fields_provided:
        new_token, expiry_time = st.session_state['jwt_key_store'].create_jwt_token( st.session_state['session_id'],
            service_name_reg, openai_key, hours_valid=expiry_hours_reg)
        
        # Store the new token in the session state
        composite_key = (service_name_reg, new_token)  # Create a composite key
        data = {
            "Expiry": expiry_time,
            "Status": "Active",
        }

        st.sidebar.success("API Key registered successfully!")
        # Reset the flag to False once the operation is done
        st.session_state['btn_action_register'] = False


def create_history_dataframe():
    data = []
    # Iterate over the jwt_keys dictionary
    session_id = st.session_state['session_id'] 
    for (service_name, token), value in st.session_state['jwt_key_store'].jwt_keys.get(session_id, {}).items():
        # Construct a dictionary for each entry
        entry = {
            'SessionId' : session_id,
            'Service': service_name,
            'Token': token
        }
        if isinstance(value, dict):
            entry.update(value)  # Add the rest of the values from the stored dictionary
        else:
            entry['API_KEY'] = _maskAPIKey(value)
        data.append(entry)

    return pd.DataFrame(data)

def msg_hit_enter():
    st.session_state['msg_hit_enter'] = True
    

def setup_body(proxy_url):
# Main Body of the application - Top section

    col1_hist = st.container()
    with col1_hist:
        st.subheader("2. Tokens", help="Use sidebar to generate tokens")
        st.button("Revoke Tokens", key="Remove_Selected_Sessions", on_click=on_button_revoke_click)
        df = create_history_dataframe()
        # Check if DataFrame is not empty
        if not df.empty:
            # Add a checkbox column to the DataFrame for selection
            df['Select'] = False
            df['Remove'] = False
            # Reorder DataFrame columns to place 'Select' and 'Remove' first
            column_order = ['Select', 'Remove'] + [col for col in df.columns if col not in ['Select', 'Remove']]
            reordered_df = df[column_order]
            # Make the DataFrame editable with checkboxes for selection
            edited_df = st.data_editor(
                reordered_df,
                column_config={
                'Select': st.column_config.CheckboxColumn (label='Select', help ="select the token to be used"),
                'Remove': st.column_config.CheckboxColumn(label='Remove')
                },
                use_container_width=True
            )

            selected_rows = edited_df[edited_df['Select'] == True]
            if not selected_rows.empty:
                selected_token = selected_rows.iloc[0]['Token']
                st.session_state['selected_token'] = selected_token
            else: st.session_state['selected_token'] = ""    


            if st.session_state.get('btn_action_revoke', False):
                st.session_state['btn_action_revoke'] = False
                deleted= False
                for index, row in edited_df.iterrows():
                    if row['Remove']:
                        sessionId = row['SessionId']
                        service   = row['Service']
                        token     =  row['Token']
                        if (service, token) in st.session_state['jwt_key_store'].jwt_keys[sessionId]:
                            del st.session_state['jwt_key_store'].jwt_keys[sessionId][(service, token)]
                            deleted = True
                if deleted:
                    st.rerun()


    st.markdown("---")  # Separator

    code1 = st.container()
    exc_btn_clicked = False
    with code1 :    
        st.subheader("3. Playground: :green[ Simulate API Calls]", help="Input your question, click Execute")
        msg = st.text_input("Input your Question", "Say, This is an API test completed using Token", key="input_msg", on_change=msg_hit_enter)
        formatted_code = show_code(sessionid=st.session_state['session_id'],token=st.session_state["selected_token"], proxy_url=proxy_url, msg=msg)
        if st.session_state.input_msg and st.session_state['msg_hit_enter'] :
            exc_btn_clicked = True
            if st.session_state["selected_token"]=="":
             st.error("Please select a token to be used")
            else :
                with st.spinner("Accessing OpenAI Through the Inbuilt Proxy Service"):
                    process, output_queue, error_queue = execute_code_in_process(formatted_code)
                    process.join()  # Wait for the process to finish
                    
                # Check for errors first
                if not error_queue.empty():
                    error = error_queue.get()
                    st.error(f"Error in user code: {error}")
                    st.session_state["out_put"] =error
                    st.write(f":red[{st.session_state['out_put']}]")
                # Check for standard output
                elif not output_queue.empty():
                    output = output_queue.get().strip()
                    st.session_state["out_put"] =output
                    output_dict = ast.literal_eval(st.session_state['out_put'])
                    st.json(output_dict)    


    st.markdown("---")  # Separator

    col2_log = st.container()
    with col2_log:
        st.subheader("4. API call History", help="shows call activities")
        if exc_btn_clicked or st.session_state['history_data_loaded']:
            log_data = load_log_data()
            df = pd.DataFrame(log_data)
            st.dataframe(df, use_container_width=True)

    st.markdown("---")  # Separator

    st.subheader("5. Reference Architecture")
    with st.expander("Flow Diagram"):
        show_image()
    

def run_fastapi_proxy(fproxy):
    fproxy.run()


@st.cache_resource
def getFastAPIProxy_instance():
    return FastAPIProxy.instance()

def how_to_use():
    # 'How To' Section
    with st.expander("How to Section"):
        st.markdown("""
        **A Demo Application - Do Not Use in Production**

        Follow these simple steps to interact with the application:

        1. **Generate a Token:** 
        In the sidebar, enter your OpenAI API key in the "OpenAI API Key *" input field. This action will generate a token for you.

        2. **Select a Token:** 
        Choose the desired token from the list of Tokens by clicking on the corresponding checkbox.

        3. **Query OpenAI:** 
        Head over to the "Playground" section. Here, type in your query for OpenAI and simply hit the 'Enter' key to execute. Ensure you have selected a token before submitting your query. This feature mimics the functionality of a third-party application.

        4. **Reference Architecture:** 
        Curious about the underlying structure? Scroll down to the "Reference Architecture" at the bottom of the page for more insights. This Streamlit application integrates a FastAPI proxy and operates on the same server as the Streamlit app itself.
        """)


def main():
    
    st.set_page_config(layout="wide", page_title="BreifKey")
    st.markdown(f'<h1 style="color:green; text-align: center;">BreifKey: Making OpenAI API Use Easier and Safer</h1>', unsafe_allow_html=True)
    how_to_use()
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = generate_session_id()
        st.session_state["selected_token"] = ""

    if 'msg_hit_enter' not in  st.session_state:
        st.session_state['msg_hit_enter'] = False

    if 'history_data_loaded' not in  st.session_state:
        st.session_state['history_data_loaded'] = False


    if 'out_put' not in  st.session_state:
         st.session_state[''] = False
         st.session_state["out_put"] = "No results show"

   
    if 'hostname' not in st.session_state:
        st.session_state['hostname'] = platform.uname()[1]

    # Add a flag to track if the FastAPI server has been started
    fastapi_thread = find_fastapi_server_thread()
    fproxy = getFastAPIProxy_instance()
    if 'fastapi_proxy' not in st.session_state:
        st.session_state['fastapi_proxy'] = fproxy

    if 'jwt_key_store' not in st.session_state:
        st.session_state['jwt_key_store'] = fproxy.jwt_key_store_ref
        
    proxy_url = f"http://{fproxy.config_host}:{fproxy.config_port}"        
    
    if not fastapi_thread:
        st.session_state['fastapi_thread'] = threading.Thread(target=run_fastapi_proxy, args=(getFastAPIProxy_instance(),), daemon=True, name=unique_thread_name)
        st.session_state['fastapi_thread'].start()
        
    setupsidebar_registration(proxy_url=proxy_url)
    setup_body(proxy_url=proxy_url)

if __name__ == "__main__":
    main()
