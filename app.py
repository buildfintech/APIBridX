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

def _maskAPIKey(key):
    return '*' * (4) + key[-4:] if len(key)> 4 else '*'

unique_thread_name = "_fastapri_proxy_thread_88765"

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
                cls._instance = super(JWTKeyStore, cls).__new__(cls)
                # Now jwt_keys is a dict of dicts, with the top-level key being the session ID
                cls._instance.jwt_keys = {}
                cls._instance.JWT_SECRET = '_Thentral_NAM8Bker#V*)'
                cls._instance.JWT_ALGORITHM = 'HS256'
        return cls._instance

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls.__new__(cls)
        return cls._instance
    
    # Client side call
    def create_jwt_token(self, session_id, service_name, openai_key, hours_valid=24):
        expiry_time = datetime.datetime.utcnow() + datetime.timedelta(hours=hours_valid)
        payload = {
            "service_name": service_name,
            "exp": expiry_time
        }
        token = jwt.encode(payload, self.JWT_SECRET, algorithm=self.JWT_ALGORITHM)

        with self._lock:
            if session_id not in self.jwt_keys:
                self.jwt_keys[session_id] = {}  # Initialize a new dict for the session if it doesn't exist
            self.jwt_keys[session_id][(service_name, token)] = openai_key

        return token, expiry_time
    
    # Server side call
    def get_service_key(self, session_id, service_name, token):
        with self._lock:
            # Retrieve the inner dict using the session_id and then the openai_key using the service_name and token
            return self.jwt_keys.get(session_id, {}).get((service_name, token))
        
    # Server side call
    def is_valid_session(self, session_id):
        with self._lock:
            # check key exist
            return session_id in self.jwt_keys


    # Server side call    
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
                cls._instance.initialized = False  # Indicates if instance is initialized
        return cls._instance

    @classmethod
    def instance(cls, host, jwtkeystore_ref):
        inst = cls.__new__(cls)
        if not inst.initialized:
            inst.__init__(host, jwtkeystore_ref)
            inst.initialized = True
        return inst

    def __init__(self, host, jwtkeystore_ref: JWTKeyStore):
        if not self.initialized:
            self.host = host
            self.app = FastAPI()
            self.logs = {}
            self.jwt_key_store_ref = jwtkeystore_ref
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
            config = uvicorn.Config(self.app, host=self.host, port=8000)
            server = uvicorn.Server(config)
            print(f"server started {server}")
            server.run()
        except Exception as e:
            SystemError("FastAPI server failed to start")

def load_log_data():
    # URL of the FastAPI logs endpoint
    hostname = st.session_state['hostname']
    session_id = st.session_state['session_id']
    url = "http://{host}:{port}/logs".format(host=hostname, port=8000)
    response = requests.get(url,  params={'session_id': session_id})
    if response.status_code == 200:
        return response.json()
    else:
        return ["No Activites found"]

def show_code():
    code = '''
    #1. Form a request
    request_data = {
        "original_url": "https://api.openai.com/v1/chat/completions",
        "original_body": {
            "model": "gpt-3.5-turbo-1106",  # Specify the model
            "messages": [{"role": "user", "content": "Hi, This is a test"}]
        }
    }
    #2. API call routed via Proxy server
    response = requests.post(
        "http://proxy_server:8000/proxy",
        headers={
            "Authorization": f"{token}", "Service-Name": service_name, 
            "Session_Id": session_id, },
        json=request_data 
    )
    '''
    st.code(code, language='python')    

def show_image():
    st.image('TokenFlow.jpg')

def on_button_register_click():
    st.session_state['btn_action_register'] = True

def on_button_remove_selected_click():
    st.session_state['btn_action_remove_selected'] = True


def setupsidebar_registration(fproxy, session_id):
    with st.sidebar:
        st.write(f":red[Not for Production use]")
        st.write(f"Session ID: :blue[{session_id}]")
        # FastAPIProxy running status
        proxy_status = "Running" if find_fastapi_server_thread()  else "Stopped"
        color = ":green" if proxy_status == "Running" else ":red"
        st.write(f"Proxy: {color}[{proxy_status}]")
        
        st.title("Register Keys")

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
        new_token, expiry_time = st.session_state['jwt_keys_store'].create_jwt_token( st.session_state['session_id'],
            service_name_reg, openai_key, hours_valid=expiry_hours_reg)
        
        # Store the new token in the session state
        composite_key = (service_name_reg, new_token)  # Create a composite key
        data = {
            "Expiry": expiry_time,
            "Status": "Active",
        }

        st.success("API Key registered successfully!")
        # Reset the flag to False once the operation is done
        st.session_state['btn_action_register'] = False


def create_history_dataframe():
    data = []
    # Iterate over the jwt_keys dictionary
    session_id = st.session_state['session_id'] 
    for (service_name, token), value in st.session_state['jwt_keys_store'].jwt_keys.get(session_id, {}).items():
        # Construct a dictionary for each entry
        entry = {
            'Session Id' : session_id,
            'Service': service_name,
            'Token': token
        }
        if isinstance(value, dict):
            entry.update(value)  # Add the rest of the values from the stored dictionary
        else:
            entry['API_KEY'] = _maskAPIKey(value)
        data.append(entry)

    return pd.DataFrame(data)


def setup_body():
# Main Body of the application - Top section
    col1_hist = st.container()
    with col1_hist:
        st.button("Revoke Tokens", key="Remove_Selected_Sessions", on_click=on_button_remove_selected_click)
        df = create_history_dataframe()
        # Check if DataFrame is not empty
        if not df.empty:
            # Add a checkbox column to the DataFrame for selection
            df['Remove'] = False
            # Make the DataFrame editable with checkboxes for selection
            edited_df = st.data_editor(
                df,
                column_config={
                    'Remove': st.column_config.CheckboxColumn(label='Remove')
                }
            )
            if st.session_state.get('btn_action_remove_selected', False):
                st.session_state['btn_action_remove_selected'] = False            
                keys_to_remove = [(row['Service'], row['Token']) for index, row in edited_df.iterrows() if row['Remove']]
                st.session_state['jwt_keys_store'].jwt_keys = {key: value for key, value in st.session_state['jwt_keys_store'].jwt_keys.items() if key not in keys_to_remove}
                st.rerun()

    col2_log = st.container()
    with col2_log:
        if st.button("Refresh") or 1:
            # log_data = load_log_data()
            df = pd.DataFrame(["No data"])
            st.subheader("API call History")
            st.dataframe(df)

def run_fastapi_proxy(fproxy):
    fproxy.run()


@st.cache_resource
def getFastAPIProxy_instance():
    return FastAPIProxy.instance(st.session_state['hostname'],st.session_state['jwt_keys_store'] )

def main():
    
    st.set_page_config(layout="wide", page_title="APIBridX")
    st.markdown(f'<h1 style="color:blue; text-align: center;">APIBridX Token Generator - Demo</h1>', unsafe_allow_html=True)

    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = generate_session_id()

   
    if 'jwt_keys_store' not in st.session_state:
        st.session_state['jwt_keys_store'] = JWTKeyStore.instance()

    if 'hostname' not in st.session_state:
        st.session_state['hostname'] = platform.uname()[1]

    # Add a flag to track if the FastAPI server has been started
    fastapi_thread = find_fastapi_server_thread()
    if 'fastapi_proxy' not in st.session_state:
        st.session_state['fastapi_proxy'] = getFastAPIProxy_instance()

    if not fastapi_thread:
        st.session_state['fastapi_thread'] = threading.Thread(target=run_fastapi_proxy, args=(getFastAPIProxy_instance(),), daemon=True, name=unique_thread_name)
        st.session_state['fastapi_thread'].start()
        

    setupsidebar_registration(st.session_state['fastapi_proxy'], st.session_state['session_id'])
    st.subheader("Registered API Keys")
    setup_body()
    image1, code2 = st.columns(2)
    with image1:
        st.subheader("Flow Diagram")
        show_image()
    with code2 :    
        st.subheader("Sample code")
        show_code()

if __name__ == "__main__":
    main()
