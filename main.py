# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
#
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
#
# Returns:
#
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition
from flask import Flask, jsonify, request, redirect

from sklearn.metrics import accuracy_score
import numpy as np
import json  #JSON5 MODULE
import os
import psutil
import datetime
import threading
import socket
import math
from utils import append_df_to_excel

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
path = 'ImagesAttendance'

classNamesList = []
encodeList = []
myFaceAttendanceList = os.listdir(path)

#classNamesList.append(['Barak Obama'])
#classNamesList.append(['Raja Sharma'])


def list_AttendanceDir(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file and allowed_file(file):
                r.append(os.path.join(root, file))
    return r


def write_neuralJson(neuralJson):
    #Writing JSON to a file
    # person_dict = {"name": "Bob",
    # "languages": ["English", "Fench"],
    # "married": True,
    # "age": 32
    # }

     with open('neural.json', 'w') as json_file:
        json.dump(neuralJson, json_file)

    #print(neuralJson)

def list_filesDir(myList):
    neural_json = []
    for cl in myList:
        neural = []
        i = 1
        label = os.path.splitext(cl)[0]
        myListF = list_files(f'{path}/{cl}')
        for clf in myListF:
            img = face_recognition.load_image_file(clf)
            if len(myListF) == 1:
                neural.append(np.array(face_recognition.face_encodings(img)[0]))
            else:
                neural.append(np.array(face_recognition.face_encodings(img)[0]))

        descriptors = np.array(neural).tolist()
        neural_json.append({"label": label, "descriptors": descriptors})

    #print(neural_json)
    write_neuralJson(neural_json)


#list_filesDir(myFaceAttendanceList)

def load_neuralinfo():
    with open('neural.json') as f:
      neuraldata = json.load(f)

      if neuraldata[0] == []:
          print
          'No Data!'
      else:
          # Clearing neural array list
          classNamesList.clear()
          encodeList.clear()
          for rows in neuraldata:
              for facedata  in rows['descriptors']:
                  classNamesList.append(rows['label'])
                  encodeList.append(facedata)
                  #print(classNamesList)
                  # print(encodeList)
          print("neural info update done...")


def markAttendance(empid, name, timestamp):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            #now = datetime.now()
            #dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{empid},{name},{timestamp}')


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return detect_faces_in_image(file)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>Face Recognition?</title>
    <style>body{font-family: 'Segoe UI'; font-size: 12pt;}
		header h1{font-size:12pt; color: #fff; background-color: #1BA1E2; padding: 20px;}
	</style>    
    <header>
    <h1>File API – Face Recognition Upload</h1>
    </header>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''

def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

def detect_faces_in_image(file_stream):
    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
    known_face_encoding1 = [-0.09634063, 0.12095481, -0.00436332, -0.07643753, 0.0080383,
                            0.01902981, -0.07184699, -0.09383309, 0.18518871, -0.09588896,
                            0.23951106, 0.0986533, -0.22114635, -0.1363683, 0.04405268,
                            0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
                            0.03416885, -0.00267565, 0.09203379, 0.04713435, -0.12731361,
                            -0.35371891, -0.0503444, -0.17841317, -0.00310897, -0.09844551,
                            -0.06910533, -0.00503746, -0.18466514, -0.09851682, 0.02903969,
                            -0.02174894, 0.02261871, 0.0032102, 0.20312519, 0.02999607,
                            -0.11646006, 0.09432904, 0.02774341, 0.22102901, 0.26725179,
                            0.06896867, -0.00490024, -0.09441824, 0.11115381, -0.22592428,
                            0.06230862, 0.16559327, 0.06232892, 0.03458837, 0.09459756,
                            -0.18777156, 0.00654241, 0.08582542, -0.13578284, 0.0150229,
                            0.00670836, -0.08195844, -0.04346499, 0.03347827, 0.20310158,
                            0.09987706, -0.12370517, -0.06683611, 0.12704916, -0.02160804,
                            0.00984683, 0.00766284, -0.18980607, -0.19641446, -0.22800779,
                            0.09010898, 0.39178532, 0.18818057, -0.20875394, 0.03097027,
                            -0.21300618, 0.02532415, 0.07938635, 0.01000703, -0.07719778,
                            -0.12651891, -0.04318593, 0.06219772, 0.09163868, 0.05039065,
                            -0.04922386, 0.21839413, -0.02394437, 0.06173781, 0.0292527,
                            0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486,
                            0.01428208, -0.03637431, 0.03971229, 0.13983178, -0.23006812,
                            0.04999552, 0.0108454, -0.03970895, 0.02501768, 0.08157793,
                            -0.03224047, -0.04502571, 0.0556995, -0.24374914, 0.25514284,
                            0.24795187, 0.04060191, 0.17597422, 0.07966681, 0.01920104,
                            -0.01194376, -0.02300822, -0.17204897, -0.0596558, 0.05307484,
                            0.07417042, 0.07126575, 0.00209804]

        # encodeList =[]
        # encodeList.append(known_face_encoding1)
        # encodeList.append(known_face_encoding2)
    # Load the uploaded image file
    img = face_recognition.load_image_file(file_stream)
    # Get face encodings for any faces in the uploaded image
    unknown_face_encodings = face_recognition.face_encodings(img)

    #getFaceNeural = face_recognition.face_encodings(img)
    # print(getFaceNeural)
    face_found = False
    face_match = False

    empid = ""
    name = ""
    match_resultsDistense = ""
    timestamp = ""
    result = ""
    if len(unknown_face_encodings) > 0:
        face_found = True

        # See if the first face in the uploaded image matches the known face of Obama
        #match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings[0])
        #tolerance (0.6 for normal cutoff) & (0.5 for very strict cutoff)
        unknown_face = unknown_face_encodings[0]
        match_results = face_recognition.compare_faces(encodeList, unknown_face, tolerance=0.5)
        match_resultsDis = face_recognition.face_distance(encodeList, unknown_face)

        matchIndex = np.argmin(match_resultsDis)
        # print(matchIndex)
        # print(classNamesList)
        # If a match was found in known_face_encodings, just use the first one.
        MAX_DISTANCE = 0.6  # increase to make recognition less strict, decrease to make more strict
        if np.any(match_resultsDis <= MAX_DISTANCE):
            best_match_idx = np.argmin(match_resultsDis)
            print("known face name : ", classNamesList[best_match_idx])
        else:
            print("known face name : not found")

        known_face_distance = ""
        if True in match_results:

            #matchtrue = true
            #face_match_percentage = (1 - match_resultsDis) * 100
            for i, face_distance in enumerate(match_resultsDis):
                #known_face_distance = format(face_distance, i)
                print("raja conf : ", face_distance_to_conf(face_distance, i))
                print("The test image has a distance of {:.2} from known image #{}".format(face_distance, i))
                #print("- With a normal cutoff, comparing with a tolerance of 0.6? {}".format(face_distance < 0.6))
                #print("- With a very strict cutoff, comparing with a tolerance of 0.5? {}".format(face_distance < 0.5))
                #print(np.round(face_match_percentage, 3))  # upto 3 decimal places

        # print(match_results)
        # print(match_resultsDis)
        # print(matchIndex)
        # print(match_results[matchIndex])
        #print(accuracy_score(encodeList[matchIndex], unknown_face))


        #if match_results[0] == True:
        if match_results[matchIndex]:
            face_match = True
            match_resultsDistense = known_face_distance
            neuralinfo = classNamesList[matchIndex]
            neuralinfoArray = neuralinfo.split("@")
            name = neuralinfoArray[0].upper()
            empid = neuralinfoArray[1]
            dt = datetime.datetime.now()
            timestamp = dt.strftime("%Y") + "/" + dt.strftime("%m") + "/" + dt.strftime("%d") + " " + dt.strftime("%H") + ":" + dt.strftime("%M") + ":" + dt.strftime("%S")

            # Printing to log file excel
            markAttendance(empid, name, timestamp)

    # Return the result as json
    result = {"result": {
        "face_found_in_image": face_found,
        "face_match_resultsDistense": match_resultsDistense,
        "face_match": face_match,
        "empid": empid,
        "name": name,
        "timestamp": timestamp
        }}
    print(result)
    return jsonify(result)

def job():
    dt = datetime.datetime.now()
    thread_timestamp = dt.strftime("%Y") + "/" + dt.strftime("%m") + "/" + dt.strftime("%d") + " " + dt.strftime(
        "%H") + ":" + dt.strftime("%M") + ":" + dt.strftime("%S")

    load_neuralinfo()
    print("I'm working..." + thread_timestamp)

    print("The Total CPU is : ", os.cpu_count())
    # Calling psutil.cpu_precent() for 4 seconds
    print("The CPU usage is : ", psutil.cpu_percent(4))
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    threading.Timer(100.0, job).start()  # called every minute

#load_neuralinfo()
#print ip address
print((([ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")] or [[(s.connect(("8.8.8.8", 53)), s.getsockname()[0], s.close()) for s in [socket.socket(socket.AF_INET, socket.SOCK_DGRAM)]][0][1]]) + ["no IP found"])[0])

if __name__ == "__main__":
    job()
    app.run(host='0.0.0.0', port=5000, debug=True)