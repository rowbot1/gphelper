
[     UTC     ] Logs for gpapper-qzgrd4wbtj8ym6bwnzp7bu.streamlit.app/
────────────────────────────────────────────────────────────────────────────────────────
[21:15:20] 🖥 Provisioning machine...
[21:15:20] 🎛 Preparing system...
[21:15:20] ⛓ Spinning up manager process...
[21:15:23] 🚀 Starting up repository: 'gphelper', branch: 'main', main module: 'app.py'
[21:15:23] 🐙 Cloning repository...
[21:15:23] 🐙 Cloning into '/mount/src/gphelper'...

[21:15:23] 🐙 Cloned repository!
[21:15:23] 🐙 Pulling code changes from Github...
[21:15:24] 📦 Processing dependencies...

──────────────────────────────────────── uv ───────────────────────────────────────────

Using uv pip install.
Resolved 60 packages in 505ms
Downloaded 60 packages in 1.83s
Installed 60 packages in 73ms
 + altair==5.3.0
 +[2024-07-05 21:15:27.034727]  annotated-types==0.7.0
 + anyio==4.4.0
 + attrs==23.2.0
 + blinker==1.8.2
 + cachetools==5.3.3
 + certifi==2024.7.4
 + charset-normalizer==3.3.2
 + click==8.1.7
 + distro==1.9.0
 + dnspython==2.6.1
 + gitdb==4.0.11
 + gitpython==3.1.43
 + groq==0.9.0
 + h11==0.14.0
 + httpcore==1.0.5
 + httpx==0.27.0
 + idna==3.7
 + importlib-metadata==6.11.0
 + jinja2==3.1.4
 +[2024-07-05 21:15:27.034970]  jsonschema==4.22.0
 + jsonschema-specifications==2023.12.1
 + loguru==0.7.2
 + markdown-it-py==3.0.0
 + markupsafe==2.1.5
 + mdurl==0.1.2
 + numpy==1.26.4
 + packaging==23.2
 + pandas==2.2.2
 + pillow==10.4.0
 + pinecone-client==2.2.4
 + protobuf==4.25.3
 + pyarrow==16.1.0
 + pydantic==2.8.2
 + pydantic-core==2.20.1
 +[2024-07-05 21:15:27.035168]  pydeck==0.9.1
 + pygments==2.18.0
 + python-dateutil==2.9.0.post0
 + pytz==2024.1
 + pyyaml==6.0.1
 + referencing==0.35.1
 + requests==2.32.3
 + rich==13.7.1
 + rpds-py==0.18.1
 + six==1.16.0
 + smmap==5.0.1
 + sniffio==1.3.1
 + streamlit==1.28.0
 + tenacity==8.5.0
 + toml==0.10.2
 + toolz==0.12.1
 + tornado==6.4.1
 + tqdm==4.66.4
 [2024-07-05 21:15:27.035299] + typing-extensions==4.12.2
 + tzdata==2024.1
 + tzlocal==5.2
 + urllib3==2.2.2
 + validators==0.30.0
 + watchdog==4.0.1
 + zipp==3.19.2
Checking if Streamlit is installed
Found Streamlit version 1.28.0 in the environment

────────────────────────────────────────────────────────────────────────────────────────

[21:15:31] 🐍 Python dependencies were installed from /mount/src/gphelper/requirements.txt using uv.
Check if streamlit is installed
Streamlit is already installed
[21:15:32] 📦 Processed dependencies!




  A new version of Streamlit is available.

  See what's new at https://discuss.streamlit.io/c/announcements

  Enter the following command to upgrade:
  $ pip install streamlit --upgrade

────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
[21:20:26] ❗️ 
2024-07-05 21:20:26.160 503 GET /script-health-check (127.0.0.1) 5.53ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:20:31.278 503 GET /script-health-check (127.0.0.1) 140.12ms
2024-07-05 21:20:36.058 503 GET /script-health-check (127.0.0.1) 1.76ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:20:41.139 503 GET /script-health-check (127.0.0.1) 5.77ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:20:46.246 503 GET /script-health-check (127.0.0.1) 166.87ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:20:51.153 503 GET /script-health-check (127.0.0.1) 4.61ms
2024-07-05 21:20:56.130 503 GET /script-health-check (127.0.0.1) 1.84ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:01.346 503 GET /script-health-check (127.0.0.1) 168.37ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:06.309 503 GET /script-health-check (127.0.0.1) 114.86ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:11.302 503 GET /script-health-check (127.0.0.1) 153.13ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:16.312 503 GET /script-health-check (127.0.0.1) 169.86ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:21.291 503 GET /script-health-check (127.0.0.1) 159.28ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:26.280 503 GET /script-health-check (127.0.0.1) 128.80ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:31.064 503 GET /script-health-check (127.0.0.1) 4.80ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:36.257 503 GET /script-health-check (127.0.0.1) 169.54ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:41.072 503 GET /script-health-check (127.0.0.1) 5.09ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:46.254 503 GET /script-health-check (127.0.0.1) 52.87ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:51.168 503 GET /script-health-check (127.0.0.1) 5.30ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:21:56.094 503 GET /script-health-check (127.0.0.1) 4.53ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:01.176 503 GET /script-health-check (127.0.0.1) 54.90ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:06.262 503 GET /script-health-check (127.0.0.1) 108.37ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:11.203 503 GET /script-health-check (127.0.0.1) 128.71ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:16.265 503 GET /script-health-check (127.0.0.1) 171.20ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:21.296 503 GET /script-health-check (127.0.0.1) 168.08ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:26.335 503 GET /script-health-check (127.0.0.1) 165.09ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:31.150 503 GET /script-health-check (127.0.0.1) 104.60ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:36.146 503 GET /script-health-check (127.0.0.1) 57.61ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:41.185 503 GET /script-health-check (127.0.0.1) 53.80ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:46.194 503 GET /script-health-check (127.0.0.1) 142.85ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:51.132 503 GET /script-health-check (127.0.0.1) 4.37ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:22:56.133 503 GET /script-health-check (127.0.0.1) 4.35ms
2024-07-05 21:23:01.143 503 GET /script-health-check (127.0.0.1) 1.54ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:23:06.359 503 GET /script-health-check (127.0.0.1) 168.21ms
[21:23:10] 🐙 Pulling code changes from Github...
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:3 in <module>                                      
                                                                                
     1 import os                                                                
     2 import streamlit as st                                                   
  ❱  3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
     5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
────────────────────────────────────────────────────────────────────────────────
ImportError: cannot import name 'Pinecone' from 'pinecone' 
(/home/adminuser/venv/lib/python3.11/site-packages/pinecone/__init__.py)
2024-07-05 21:23:11.124 503 GET /script-health-check (127.0.0.1) 4.99ms
[21:23:11] 📦 Processing dependencies...
[21:23:11] 📦 Processed dependencies!
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 from pinecone import Pinecone                                            
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7 from dotenv import load_dotenv                                           
     8                                                                          
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
[21:23:13] 🔄 Updated app!
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:16.258 503 GET /script-health-check (127.0.0.1) 159.40ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:21.341 503 GET /script-health-check (127.0.0.1) 185.97ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:26.307 503 GET /script-health-check (127.0.0.1) 162.99ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:31.314 503 GET /script-health-check (127.0.0.1) 168.09ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:36.292 503 GET /script-health-check (127.0.0.1) 192.62ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:41.294 503 GET /script-health-check (127.0.0.1) 154.48ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:46.356 503 GET /script-health-check (127.0.0.1) 200.07ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:51.307 503 GET /script-health-check (127.0.0.1) 187.25ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:23:56.291 503 GET /script-health-check (127.0.0.1) 181.29ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:01.349 503 GET /script-health-check (127.0.0.1) 145.22ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:06.279 503 GET /script-health-check (127.0.0.1) 176.00ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:11.265 503 GET /script-health-check (127.0.0.1) 163.90ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:16.135 503 GET /script-health-check (127.0.0.1) 5.24ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:21.350 503 GET /script-health-check (127.0.0.1) 203.49ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:26.307 503 GET /script-health-check (127.0.0.1) 172.10ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:31.334 503 GET /script-health-check (127.0.0.1) 183.96ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:36.310 503 GET /script-health-check (127.0.0.1) 199.05ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:41.286 503 GET /script-health-check (127.0.0.1) 174.91ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:46.401 503 GET /script-health-check (127.0.0.1) 208.22ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:51.323 503 GET /script-health-check (127.0.0.1) 196.69ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:24:56.218 503 GET /script-health-check (127.0.0.1) 156.54ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:25:01.408 503 GET /script-health-check (127.0.0.1) 205.71ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:25:06.304 503 GET /script-health-check (127.0.0.1) 196.22ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:25:11.266 503 GET /script-health-check (127.0.0.1) 196.25ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:25:16.303 503 GET /script-health-check (127.0.0.1) 208.90ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:25:21.299 503 GET /script-health-check (127.0.0.1) 197.47ms
────────────────────── Traceback (most recent call last) ───────────────────────
  /home/adminuser/venv/lib/python3.11/site-packages/streamlit/runtime/scriptru  
  nner/script_runner.py:534 in _run_script                                      
                                                                                
  /mount/src/gphelper/app.py:5 in <module>                                      
                                                                                
     2 import streamlit as st                                                   
     3 import pinecone                                                          
     4 from groq import Groq                                                    
  ❱  5 from sentence_transformers import SentenceTransformer                    
     6 import numpy as np                                                       
     7                                                                          
     8 # Initialize Pinecone                                                    
────────────────────────────────────────────────────────────────────────────────
ModuleNotFoundError: No module named 'sentence_transformers'
2024-07-05 21:25:26.237 503 GET /script-health-check (127.0.0.1) 162.94ms
