from flask import Flask
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'oiawfansf345rnfwsk4'
socketio = SocketIO(app)

if __name__ == '__main__':
    socketio.run(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"