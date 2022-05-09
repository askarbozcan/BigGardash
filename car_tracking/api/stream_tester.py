import requests
import cv2

if __name__ == "__main__":
    # check what localhost:4920/stream responds with
    
    r = requests.get("http://localhost:4920/stream", stream=True)

    for line in r.iter_lines():
        if b"--frame" in line:
            continue
            
        print(line)
        print("==========================")