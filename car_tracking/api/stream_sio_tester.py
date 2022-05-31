import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')
    sio.emit("give_stream_data")

@sio.event
def receive_stream_data(data):
    print("started receiving stream data")
    print(data.keys())
    #sio.wait()
if __name__ == "__main__":
    sio.connect("http://localhost:4920")
    sio.wait()