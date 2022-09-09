# Intent chatbot websocket server example for library application
Backend that delivers the response from the ML model via websocket, see [client](https://github.com/mal2/python-chatbot-client)
## Setup
Creating a virtual environment and activate it:
```
virtualenv -p python3.9 venv
```
Activate for linux
```
source venv/bin/activate
```
or for Windows:
```
.\venv\Scripts\activate
```
Then install the required python librarys using:
```
pip install -r requirements.txt
```
## Usage
Finally run by using:
```
uvicorn run:app --reload --port 5000
```
