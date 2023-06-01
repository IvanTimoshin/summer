import cv2
import numpy as np
import os
import glob
import struct
import pickle
import socket
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
safeCoords = []
def get_img():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '192.168.0.11'
    port = 9988
    client_socket.connect((host_ip, port))

    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        client_socket.close()
        return frame
    
while True:
    img = get_img()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.aruco.detectMarkers(gray,dictionary)
    coords = []
    if res[1] is not None and 0 in res[1] and 1 in res[1] and 2 in res[1] and 3 in res[1]:
        for marker in range(4):
            index = np.where(res[1] == marker)[0][0]
            pt0 = res[0][index][0][marker].astype(np.int16)
            coords.append(list(pt0))
            cv2.circle(img, pt0, 10, (255,255,255), thickness=-1)
        with open("save.coords","wb") as f:
            np.save(f, coords)
        safeCoords = coords
        height, width, _ = img.shape
        input_pt = np.array(coords)
        output_pt = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        h, _ = cv2.findHomography(input_pt, output_pt)
        res_img = cv2.warpPerspective(img, h, (width, height))
        cv2.imshow('111',res_img)
    elif res[1] is not None and safeCoords is not None:
        with open("save.coords","rb") as f:
            safeCoords=np.load(f)
        for i in range(4):
            cv2.circle(img, safeCoords[i], 10, (0,255,255), thickness=-1)
        height, width, _ = img.shape
        input_pt = np.array(safeCoords)
        output_pt = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        h, _ = cv2.findHomography(input_pt, output_pt)
        res_img = cv2.warpPerspective(img, h, (width, height))
        cv2.imshow('111',res_img)
    cv2.imshow('img',img)
    if cv2.waitKey(5) == 27:
        break
